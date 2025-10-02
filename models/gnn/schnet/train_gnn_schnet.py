# -*- coding: utf-8 -*-
"""
train_gnn_schnet.py — 轻量 SchNet 风格 GNN（支持蒸馏到 run4 老师）
依赖：torch, ase, numpy, pandas, tqdm, scikit-learn（用于metrics）
"""

import os, json, math, argparse, io, sys, glob, random
import numpy as np
import pandas as pd
from tqdm import tqdm

# 控制台 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ase import Atoms

def teacher_predict_batch_from_tensor(pipe, soap, pooling, Z_batch, pos_batch, device):
    """给一批 (Z,pos) 张量，用老师（SOAP+KRR）做预测；返回 torch.float32, shape [B]"""
    feats = []
    Z_np = Z_batch.detach().cpu().numpy()
    pos_np = pos_batch.detach().cpu().numpy()
    for b in range(len(Z_np)):
        at = Atoms(numbers=Z_np[b].astype(int), positions=pos_np[b])
        feats.append(structure_soap_vector(soap, at, pooling=pooling))
    X = np.stack(feats, axis=0)
    yhat = pipe.predict(X)
    return torch.tensor(yhat, dtype=torch.float32, device=device)

# 数学加速与 CUDA 优化
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# 复用你之前的几何/描述子工具（同目录下已有）
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # models/gnn -> models
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # models -> project root
# --- add project root to sys.path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))                       # .../models/gnn/schnet
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir))  # .../DataCompetition
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.baselines.soap_krr.dataset import load_xyz_from_file
from models.baselines.soap_krr.features import build_soap_descriptor, structure_soap_vector
from models.baselines.soap_krr.model import regression_metrics

# ----------------- 实用函数 -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pairwise_distances(x):  # [N,3] -> [N,N]
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # [N,N,3]
    return torch.linalg.norm(diff, dim=-1), diff  # (d_ij, r_i - r_j)

def radial_basis(d, centers, gamma):
    # d: [E], centers: [K] ; returns [E,K]
    return torch.exp(-gamma * (d.unsqueeze(-1) - centers) ** 2)

# ----------------- 模型 -----------------
class CFConv(nn.Module):
    """连续滤波卷积（SchNet 核心），不显式用方向，得到 E(3)-不变消息"""
    def __init__(self, hidden_dim=128, n_rbf=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lin_in = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 作用在 h_j 上

    def forward(self, h, rbf, edge_index):
        # 统一精度：h 与 rbf 至少有一个会在 AMP 下变成 Half
        if h.dtype != rbf.dtype:
            # 因为在外层我们已把 h cast 到 rbf.dtype，这里再兜底一次
            h = h.to(rbf.dtype)

        src, dst = edge_index  # j->i

        w = self.edge_mlp(rbf)             # [E,D]（在 AMP 下多半是 Half）
        m = w * self.lin_in(h[src])        # [E,D]

        # ★ 关键：index_add_ 要求 self 和 source 同 dtype
        if m.dtype != h.dtype:
            m = m.to(h.dtype)

        agg = torch.zeros(h.size(0), h.size(1), dtype=h.dtype, device=h.device)
        dst = dst.to(h.device)
        agg.index_add_(0, dst, m)          # 聚合到 i

        out = self.node_mlp(agg)
        if out.dtype != h.dtype:
            out = out.to(h.dtype)

        return h + out                     # 残差

class SchNetBlock(nn.Module):
    def __init__(self, hidden_dim=128, n_rbf=64):
        super().__init__()
        self.cf = CFConv(hidden_dim, n_rbf)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, rbf, edge_index):
        h = self.cf(h, rbf, edge_index)
        return self.norm(h)

class SchNetEnergy(nn.Module):
    def __init__(self, hidden_dim=128, n_blocks=4, n_rbf=64, rcut=6.0, Z=79):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_rbf = n_rbf
        self.rcut = rcut
        # 只有 Au 也用通用 embedding，方便以后多元素扩展
        self.embed = nn.Embedding(100, hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        centers = torch.linspace(0.0, rcut, n_rbf)
        self.register_buffer("rbf_centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0 / (rcut**2)), requires_grad=False)

        self.blocks = nn.ModuleList([SchNetBlock(hidden_dim, n_rbf) for _ in range(n_blocks)])
        self.readout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    # —— 单图构边（兼容老路径）——
    def build_graph_single(self, pos, rcut=None):
        N = pos.size(0)
        if rcut is None:
            src, dst = torch.triu_indices(N, N, offset=1, device=pos.device)
            edge_index = torch.stack([src, dst], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            return edge_index
        # rcut 邻接
        d = torch.cdist(pos, pos)  # [N,N]
        mask = (d > 0) & (d <= rcut)
        src, dst = torch.nonzero(mask, as_tuple=True)
        if src.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=pos.device)
        return torch.stack([src, dst], dim=0)

    # —— 批量构边（B 个图拼一起，不跨图连边）——
    def build_graph_batch(self, pos, rcut=None):
        # pos: [B,N,3]
        B, N, _ = pos.shape
        edges = []
        for b in range(B):
            eb = self.build_graph_single(pos[b], rcut=rcut)
            if eb.numel() > 0:
                # 索引偏移到全局节点编号区间
                offset = b * N
                edges.append(eb + offset)
        if len(edges) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=pos.device)
        return torch.cat(edges, dim=1)  # [2, E_total]

    def edge_features(self, pos_flat, edge_index):
        # pos_flat: [M,3], edge_index: [2,E]
        src, dst = edge_index
        d = torch.linalg.norm(pos_flat[src] - pos_flat[dst], dim=-1)  # [E]
        d = torch.clamp(d, max=self.rcut)
        rbf = torch.exp(-self.gamma * (d.unsqueeze(-1) - self.rbf_centers) ** 2)  # [E,K]
        return rbf

    def forward(self, Z, pos, edge_index=None):
        """
        支持：
          - 单图：Z:[N], pos:[N,3]    -> 返回标量 E
          - 批量：Z:[B,N], pos:[B,N,3] -> 返回 [B] 的能量向量
        """
        is_batched = (pos.dim() == 3)
        if is_batched:
            B, N, _ = pos.shape
            Z_flat  = Z.reshape(B * N)
            pos_flat = pos.reshape(B * N, 3)
            h = self.embed(Z_flat)  # [B*N, D]
            if edge_index is None:
                edge_index = self.build_graph_batch(pos, rcut=self.rcut)  # [2,E]
            rbf = self.edge_features(pos_flat, edge_index)  # [E,K]
        else:
            N = pos.size(0)
            h = self.embed(Z)  # [N,D]
            if edge_index is None:
                edge_index = self.build_graph_single(pos, rcut=self.rcut)  # [2,E]
            rbf = self.edge_features(pos, edge_index)  # [E,K]

        # —— AMP 下 dtype 对齐：节点特征与边特征保持一致 ——
        if h.dtype != rbf.dtype:
            h = h.to(rbf.dtype)

        # 通过若干 CFConv 块（按全局拼图处理）
        for blk in self.blocks:
            h = blk(h, rbf, edge_index)  # h: [M,D], M=N 或 B*N

        # 原子能量 -> 结构能
        e_atom = self.readout(h).squeeze(-1)  # [M]
        if e_atom.dtype != h.dtype:
            e_atom = e_atom.to(h.dtype)

        if is_batched:
            # 把每个图的 N 个原子能量相加：segment sum (不用 torch_scatter)
            graph_idx = torch.arange(B, device=e_atom.device).repeat_interleave(N)  # [B*N]
            E = torch.zeros(B, dtype=e_atom.dtype, device=e_atom.device)
            E.index_add_(0, graph_idx, e_atom)  # [B]
            return E
        else:
            return e_atom.sum()

# ----------------- 数据集 -----------------
class Au20Dataset(Dataset):
    def __init__(self, ids, energies_df, xyz_dir, outlier_weights=None):
        self.ids = [str(i) for i in ids]
        self.df = energies_df.set_index("id", drop=False)
        self.xyz_dir = xyz_dir
        self.outlier_w = None
        if outlier_weights is not None:
            self.outlier_w = {k: float(v) for k,v in outlier_weights.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        fp = os.path.join(self.xyz_dir, f"{iid}.xyz")
        atoms_list, _ = load_xyz_from_file(fp)
        assert len(atoms_list) == 1, f"{iid} 多帧？"
        at = atoms_list[0]
        pos = torch.as_tensor(at.get_positions(), dtype=torch.float32)
        Z = torch.as_tensor(at.get_atomic_numbers(), dtype=torch.long)

        y = float(self.df.loc[iid, "DeltaE"])
        w = 1.0
        if self.outlier_w is not None and iid in self.outlier_w:
            w = self.outlier_w[iid]
        return dict(id=iid, Z=Z, pos=pos, y=torch.tensor(y, dtype=torch.float32), w=torch.tensor(w, dtype=torch.float32))

def collate(batch):
    # Au20 固定原子数，简单打包
    ids = [b["id"] for b in batch]
    Z = torch.stack([b["Z"] for b in batch], dim=0)       # [B,N]
    pos = torch.stack([b["pos"] for b in batch], dim=0)   # [B,N,3]
    y = torch.stack([b["y"] for b in batch], dim=0)       # [B]
    w = torch.stack([b["w"] for b in batch], dim=0)       # [B]
    return ids, Z, pos, y, w

# ----------------- 老师（run4）蒸馏工具 -----------------
def load_teacher(teacher_dir):
    import joblib
    meta = json.load(open(os.path.join(teacher_dir, "meta.json"), "r", encoding="utf-8"))
    pipe = joblib.load(os.path.join(teacher_dir, "model.joblib"))
    soap = build_soap_descriptor(
        rcut=float(meta["soap"]["rcut"]),
        nmax=int(meta["soap"]["nmax"]),
        lmax=int(meta["soap"]["lmax"]),
        sigma=float(meta["soap"]["sigma"]),
        species=("Au",)
    )
    pooling = meta["soap"].get("pooling", "mean")
    return pipe, soap, pooling

def teacher_predict_ids(pipe, soap, pooling, ids, xyz_dir):
    X = []
    for iid in ids:
        fp = os.path.join(xyz_dir, f"{iid}.xyz")
        atoms_list, _ = load_xyz_from_file(fp)
        at = atoms_list[0]
        X.append(structure_soap_vector(soap, at, pooling=pooling))
    X = np.stack(X, axis=0)
    yhat = pipe.predict(X)
    return {iid: float(y) for iid, y in zip(ids, yhat)}

# ----------------- 训练主程序 -----------------
def parse_args():
    ap = argparse.ArgumentParser("SchNet-like GNN for Au20 ΔE (with optional distillation from run4)")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--cv_folds_json", required=True)
    ap.add_argument("--test_ids_json", required=True)
    ap.add_argument("--xyz_dir", required=True)

    ap.add_argument("--fold", type=int, default=1, help="使用第几折做验证（1-5）")
    ap.add_argument("--out_dir", required=True)

    # 模型超参
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--n_rbf", type=int, default=64)
    ap.add_argument("--rcut", type=float, default=6.0)

    # 训练超参
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)

    # 异常降权
    ap.add_argument("--use_outlier_weights", action="store_true", default=False)
    ap.add_argument("--outlier_weight", type=float, default=0.4)

    # 蒸馏
    ap.add_argument("--teacher_dir", type=str, default=None, help="run4 目录（含 model.joblib/meta.json）")
    ap.add_argument("--lambda_kd", type=float, default=0.3, help="蒸馏损失权重 λ")
    # Vicinal KD（几何微扰 + 老师软标）
    ap.add_argument("--use_vkd", action="store_true", default=False)
    ap.add_argument("--vkd_sigma", type=float, default=0.01, help="Å；微扰标准差")
    ap.add_argument("--vkd_beta", type=float, default=0.1, help="VKD 损失权重 β")
    ap.add_argument("--vkd_start_epoch", type=int, default=0, help="从第几轮开始启用 VKD（warmup）")

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    # 可选：显示每个 epoch 的 batch 级进度条
    ap.add_argument("--show_batch_bar", action="store_true", default=False)

    return ap.parse_args()

def set_device(dev):
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)
    device = set_device(args.device)
    print(f"[Device] {device}")

    df = pd.read_csv(args.energies_csv).astype({"id": str})
    with open(args.cv_folds_json, "r", encoding="utf-8") as f:
        folds = json.load(f)["folds"]
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_ids = [str(i) for i in json.load(f)["test_ids"]]

    fold = folds[args.fold - 1]
    train_ids = [str(i) for i in fold["train_ids"]]
    val_ids   = [str(i) for i in fold["val_ids"]]

    # 降权（如启用）
    w_map = None
    if args.use_outlier_weights and "is_outlier" in df.columns:
        w_map = {row.id: (args.outlier_weight if int(row.is_outlier)==1 else 1.0) for _,row in df.iterrows()}

    ds_tr = Au20Dataset(train_ids, df, args.xyz_dir, outlier_weights=w_map)
    ds_va = Au20Dataset(val_ids,   df, args.xyz_dir, outlier_weights=w_map)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  collate_fn=collate,
                       num_workers=0, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                       num_workers=0, pin_memory=(device=="cuda"))

    # 模型 + 优化器 + AMP + 调度器
    model = SchNetEnergy(hidden_dim=args.hidden_dim, n_blocks=args.n_blocks, n_rbf=args.n_rbf, rcut=args.rcut).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=20, verbose=True)

    # 老师（可选）
    kd_map = None
    if args.teacher_dir:
        pipe, soap, pooling = load_teacher(args.teacher_dir)
        kd_ids = train_ids + val_ids
        print(f"[Teacher] 计算老师预测… ({len(kd_ids)})")
        kd_map = teacher_predict_ids(pipe, soap, pooling, kd_ids, args.xyz_dir)

    best = dict(mae=1e9, state=None, epoch=-1)

    def run_epoch(dl, train=True, epoch=1, desc=""):
        model.train(train)
        tot_loss = tot_mae = tot_n = 0.0
        iterator = tqdm(dl, total=len(dl), leave=False, dynamic_ncols=True, desc=desc) if args.show_batch_bar else dl

        for ids, Z, pos, y, w in iterator:
            Z = Z.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            # === AMP 前向 ===
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                # 主分支点预测（批量）
                yhat = model(Z, pos, edge_index=None)  # [B]

                # 监督损失（带异常降权）
                mse = F.mse_loss(yhat, y, reduction="none")
                loss = (mse * w).mean()

                # 常规 KD（原几何，若提供老师软标）
                if train and (kd_map is not None):
                    with torch.no_grad():
                        y_teacher = torch.tensor([kd_map[i] for i in ids], dtype=torch.float32, device=device)
                    # 与 AMP 对齐 dtype，避免半精度/单精度混算警告
                    y_teacher = y_teacher.to(yhat.dtype)
                    kd = F.mse_loss(yhat, y_teacher)
                    loss = loss + args.lambda_kd * kd

                # === Vicinal KD：几何微扰 + 老师软标（可 warmup）===
                # 需要：args.use_vkd / args.vkd_sigma / args.vkd_beta / args.vkd_start_epoch 且存在老师 (pipe, soap, pooling)
                if train and args.use_vkd and (epoch >= args.vkd_start_epoch) and ('pipe' in globals()) and (
                        pipe is not None):
                    # 轻微原子坐标高斯扰动（Å）
                    noise = torch.randn_like(pos) * args.vkd_sigma
                    pos_v = pos + noise
                    # 学生在扰动几何上的预测
                    yhat_v = model(Z, pos_v, edge_index=None)  # [B]
                    # 老师在扰动几何上的软标签（批量版）
                    with torch.no_grad():
                        y_teacher_v = teacher_predict_batch_from_tensor(pipe, soap, pooling, Z, pos_v, device)
                    y_teacher_v = y_teacher_v.to(yhat_v.dtype)
                    vkd = F.mse_loss(yhat_v, y_teacher_v)
                    loss = loss + args.vkd_beta * vkd

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()

            with torch.no_grad():
                tot_loss += loss.item() * y.size(0)
                tot_mae += torch.mean(torch.abs(yhat - y)).item() * y.size(0)
                tot_n += y.size(0)

        return tot_loss / tot_n, tot_mae / tot_n

    patience = args.patience
    with tqdm(total=args.epochs, desc="Epochs", dynamic_ncols=True) as pbar:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_mae = run_epoch(dl_tr, train=True,  desc=f"train {epoch}")
            va_loss, va_mae = run_epoch(dl_va, train=False, desc=f"valid {epoch}")

            scheduler.step(va_mae)
            if va_mae < best["mae"]:
                best = dict(mae=va_mae, state=model.state_dict(), epoch=epoch)
                patience = args.patience  # 触发刷新耐心
            else:
                patience -= 1

            # 更新进度条状态
            lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(tr_mae=f"{tr_mae:.4f}", va_mae=f"{va_mae:.4f}",
                             best=f"{best['mae']:.4f}@{best['epoch']}", lr=f"{lr:.2e}")
            pbar.update(1)

            if patience <= 0:
                pbar.write("[EarlyStop]")
                break

    # 保存最优
    torch.save(best["state"], os.path.join(args.out_dir, f"model_fold{args.fold}.pt"))
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[Saved] best fold{args.fold} @ epoch {best['epoch']} (val MAE={best['mae']:.4f}) -> {args.out_dir}")

if __name__ == "__main__":
    main()
