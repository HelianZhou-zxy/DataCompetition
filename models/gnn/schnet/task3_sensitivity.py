# -*- coding: utf-8 -*-
"""
task3_sensitivity.py — Task 3: 局域扰动灵敏度分析（支持 KD-only / KD+VKD GNN）
- 对指定最低能代表结构（默认 id=350）施加局域随机位移；
- 用已训练 GNN 预测能量变化 ΔÊ，度量 |ΔÊ| 与扰动幅度的关系；
- 生成全局/按桶统计、二次关系拟合系数，以及原子级“软硬度” K_i。

产物（写到 --out_dir）：
  - sensitivity_samples.csv        逐样本：k, sigma, rmsd_global, rmsd_local, dE_pred, min_dist, chosen_indices
  - sensitivity_summary.csv        按 (k, sigma) 汇总：MAE, RMSE, means/medians, 线性/二次拟合系数与R²
  - site_stiffness.csv             原子级 K_i 与配位数 CN_i（可画“软硬度地图”）
  - （可选）若安装 matplotlib：自动输出两张图（散点 + 箱线图）

依赖：torch, numpy, pandas, tqdm, (可选 matplotlib)
"""

import os, json, argparse, sys, math, random
from typing import Tuple, List

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
from torch import nn

# 确保项目根在 sys.path，便于 import
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# 载入你已有的工具与模型定义
from models.baselines.soap_krr.dataset import load_xyz_from_file
from models.gnn.schnet.train_gnn_schnet import SchNetEnergy  # 与训练时一致的结构
# --- 强制把各种可能的输入（list / torch / numpy 标量/数组）变成 np.float64 ---
def to_float64_array(x):
    import numpy as _np
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        x = x.detach().cpu().numpy()
    x = _np.asarray(x)
    # 若是 0-d 标量，升成 1-d；同时强制成 float64，避免 dtype=object
    if x.ndim == 0:
        x = x.reshape(1)
    return x.astype(_np.float64, copy=False)

def to_float(x):
    import numpy as _np
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, (list, tuple)):
        return float(_np.asarray(x, dtype=_np.float64).ravel()[0])
    try:
        return float(x)
    except Exception:
        return float(_np.asarray(x).astype(_np.float64).ravel()[0])

# ---------------------- 几何/数值工具 ----------------------

def kabsch_align(P: np.ndarray, Q: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, float, float]:
    """
    用 Kabsch 对齐 Q 到 P，返回对齐后的 Q_aligned、全局RMSD 与（可选）局部RMSD
    P, Q: [N,3]
    mask: bool 索引（若提供，用于计算“局部 RMSD”）
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    P_c = P - P.mean(axis=0, keepdims=True)
    Q_c = Q - Q.mean(axis=0, keepdims=True)
    H = Q_c.T @ P_c
    U, S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = (Vt.T @ U.T)
    Q_rot = Q_c @ R
    Q_aligned = Q_rot + P.mean(axis=0, keepdims=True)

    diff = P - Q_aligned
    rmsd_global = float(np.sqrt((diff**2).sum(axis=1).mean()))
    if mask is None:
        rmsd_local = rmsd_global
    else:
        m = mask.astype(bool)
        rmsd_local = float(np.sqrt((diff[m]**2).sum(axis=1).mean()))
    return Q_aligned, rmsd_global, rmsd_local

def min_interatomic_distance(pos: np.ndarray) -> float:
    """返回结构的最小原子间距（排除对角）"""
    N = pos.shape[0]
    dmin = np.inf
    for i in range(N):
        d = np.linalg.norm(pos[i+1:] - pos[i], axis=1)
        if len(d) > 0:
            dmin = min(dmin, float(d.min()))
    return dmin

def sample_perturbed_positions(
    pos0: np.ndarray, k: int, sigma: float, dmin_thresh: float = 2.2, max_retry: int = 30, fix_com: bool = True,
    force_indices: List[int] = None, rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    从 pos0 采样一个扰动结构（局部 k 个原子正态位移 N(0, sigma^2 I)）
    - 若 force_indices 给定，则固定扰动这批原子；否则随机挑选
    - 约束：最小原子间距 >= dmin_thresh，否则重采样
    返回：pos_new, chosen_idx(bool mask), rmsd_global, rmsd_local, min_dist
    """
    if rng is None:
        rng = np.random.default_rng()
    N = pos0.shape[0]
    for _ in range(max_retry):
        if force_indices is None:
            chosen = rng.choice(N, size=k, replace=False)
        else:
            chosen = np.array(force_indices, dtype=int)
            assert len(chosen) == k
        mask = np.zeros(N, dtype=bool)
        mask[chosen] = True

        noise = np.zeros_like(pos0)
        noise[mask] = rng.normal(0.0, sigma, size=(k, 3))
        if fix_com:
            noise = noise - noise.mean(axis=0, keepdims=True)

        pos1 = pos0 + noise
        dmin = min_interatomic_distance(pos1)
        if dmin < dmin_thresh:
            continue  # 非物理近距，重采

        # Kabsch 对齐以计算 RMSD（全局和局部）
        _, rmsd_g, rmsd_l = kabsch_align(pos0, pos1, mask=mask)
        return pos1, mask, rmsd_g, rmsd_l, dmin

    # 超过重试上限，仍返回最后一次（标记 dmin 以便过滤）
    return pos1, mask, rmsd_g, rmsd_l, dmin

def coordination_numbers(pos: np.ndarray, cutoff: float = 3.2) -> np.ndarray:
    """最简配位数统计（<= cutoff 计为邻居）"""
    N = pos.shape[0]
    cn = np.zeros(N, dtype=float)
    for i in range(N):
        d = np.linalg.norm(pos - pos[i], axis=1)
        cn[i] = float(((d > 1e-8) & (d <= cutoff)).sum())
    return cn

# ---------------------- 模型预测 ----------------------

@torch.no_grad()
def predict_energy(model: nn.Module, Z: np.ndarray, pos_batch: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    批量预测能量：Z [N], pos_batch [B,N,3] -> yhat [B]
    你的 SchNetEnergy(model) 支持批量图（一次前向产出 [B]）
    """
    model.eval()
    Z_t = torch.as_tensor(Z, dtype=torch.long, device=device)
    X_t = torch.as_tensor(pos_batch, dtype=torch.float32, device=device)
    yhat = model(Z_t.unsqueeze(0).expand(X_t.size(0), -1), X_t, edge_index=None)  # [B]
    return yhat.detach().cpu().numpy()

# ---------------------- 主流程 ----------------------

def parse_args():
    ap = argparse.ArgumentParser("Task 3 — Sensitivity Analysis via Local Structural Perturbation")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--xyz_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)

    # 代表结构
    ap.add_argument("--base_id", default="350", help="代表性最低能结构的 ID（默认 350）")

    # 模型超参（要与训练时一致）
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--n_rbf", type=int, default=64)
    ap.add_argument("--rcut", type=float, default=6.0)

    # 网格与采样
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--sigma_list", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.08])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sigma_site", type=float, default=0.02, help="原子级 K_i 的微扰强度")
    ap.add_argument("--dmin", type=float, default=2.2)
    ap.add_argument("--max_retry", type=int, default=30)

    # 推理
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=2025)

    # 作图（可选）
    ap.add_argument("--make_plots", action="store_true", default=False)
    return ap.parse_args()

def set_device(dev):
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

def slope_zero_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """零截距最小二乘斜率与R²"""
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    xsq = (x*x).sum()
    if xsq <= 1e-12:
        return 0.0, 0.0
    a = float((x*y).sum() / xsq)
    yhat = a * x
    ss_res = float(((y - yhat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum() + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return a, r2

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    device = set_device(args.device)
    print(f"[Device] {device}")

    # ---------- 载入代表结构 ----------
    df = pd.read_csv(args.energies_csv).astype({"id": str})
    base_id = str(args.base_id)
    xyz_fp = os.path.join(args.xyz_dir, f"{base_id}.xyz")
    if not os.path.exists(xyz_fp):
        raise FileNotFoundError(f"代表结构 {base_id}.xyz 不存在：{xyz_fp}")
    atoms_list, _ = load_xyz_from_file(xyz_fp)
    at0 = atoms_list[0]
    pos0 = at0.get_positions().astype(np.float64)  # Kabsch 用 double 稳一点
    Z = at0.get_atomic_numbers().astype(np.int64)
    N = len(Z)
    cn = coordination_numbers(pos0, cutoff=3.2)

    # ---------- 载入模型 ----------
    model = SchNetEnergy(hidden_dim=args.hidden_dim, n_blocks=args.n_blocks, n_rbf=args.n_rbf, rcut=args.rcut)
    model.to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 原始结构基准能量（ΔE 基线）
    E0 = float(predict_energy(model, Z, pos0[None, :, :], device=device)[0])
    print(f"[Base] id={base_id}  predicted ΔE = {E0:.6f}")

    # ---------- 网格采样与预测 ----------
    rows = []
    rng = np.random.default_rng(args.seed)

    for k in args.k_list:
        for sigma in args.sigma_list:
            batch_pos = []
            batch_meta = []
            # 先采 R 个，合批推理
            for r in range(args.repeats):
                pos1, mask, rmsd_g, rmsd_l, dmin = sample_perturbed_positions(
                    pos0, k=k, sigma=sigma, dmin_thresh=args.dmin,
                    max_retry=args.max_retry, fix_com=True, rng=rng
                )
                batch_pos.append(pos1)
                batch_meta.append((mask, rmsd_g, rmsd_l, dmin))
            # 批量推理
            yhat = predict_energy(model, Z, np.stack(batch_pos, axis=0), device=device)
            dE = yhat - E0  # 相对原始结构
            for r, (mask, rmsd_g, rmsd_l, dmin) in enumerate(batch_meta):
                rows.append(dict(
                    base_id=base_id, k=k, sigma=sigma, repeat=r,
                    rmsd_global=rmsd_g, rmsd_local=rmsd_l, min_dist=dmin,
                    dE_pred=float(dE[r]),
                    chosen_indices=";".join(map(str, np.nonzero(mask)[0].tolist()))
                ))

    df_samples = pd.DataFrame(rows)
    df_samples.to_csv(os.path.join(args.out_dir, "sensitivity_samples.csv"), index=False)
    print(f"[Saved] sensitivity_samples.csv  ->  {args.out_dir}")

    # ---------- 汇总统计/拟合 ----------
    g = df_samples.groupby(["k", "sigma"])
    sum_rows = []
    for (k, sigma), d in g:
        y = d["dE_pred"].values
        x1 = d["rmsd_local"].values
        x2 = x1**2
        mae = float(np.mean(np.abs(y)))
        rmse = float(np.sqrt(np.mean(y**2)))
        mean_abs = float(np.mean(np.abs(y)))
        median_abs = float(np.median(np.abs(y)))
        a1, r2_1 = slope_zero_intercept(x1, np.abs(y))   # 线性 |ΔE| ≈ a1 * RMSD
        a2, r2_2 = slope_zero_intercept(x2, np.abs(y))   # 二次 |ΔE| ≈ a2 * RMSD^2
        sum_rows.append(dict(
            k=k, sigma=sigma, count=len(d),
            MAE_abs=mae, RMSE_abs=rmse, mean_abs=mean_abs, median_abs=median_abs,
            slope_linear=a1, R2_linear=r2_1, slope_quadratic=a2, R2_quadratic=r2_2
        ))
    df_summary = pd.DataFrame(sum_rows).sort_values(["k", "sigma"])
    df_summary.to_csv(os.path.join(args.out_dir, "sensitivity_summary.csv"), index=False)
    print(f"[Saved] sensitivity_summary.csv  ->  {args.out_dir}")

    # ---------- 全局“稳定性变量”K（小扰动区间，按 k 抽取） ----------
    small_sigma_mask = df_samples["sigma"] <= 0.02 + 1e-12
    df_small = df_samples[small_sigma_mask]
    K_rows = []
    for k, d in df_small.groupby("k"):
        x2 = (d["rmsd_local"].values ** 2)
        y = np.abs(d["dE_pred"].values)
        K, r2 = slope_zero_intercept(x2, y)  # |ΔE| ≈ K * RMSD_local^2
        K_rows.append(dict(k=k, K=K, K_per_atom=K/max(k,1), R2=r2, count=len(d)))
    pd.DataFrame(K_rows).to_csv(os.path.join(args.out_dir, "global_K_small_sigma.csv"), index=False)
    print(f"[Saved] global_K_small_sigma.csv  ->  {args.out_dir}")

    # ========= Per-atom stiffness K_i =========
    from tqdm import trange
    print("Per-atom stiffness K_i:", flush=True)

    # 读取基准结构
    atoms_list, _ = load_xyz_from_file(os.path.join(args.xyz_dir, "350.xyz"))
    at0 = atoms_list[0]
    Z0 = torch.as_tensor(at0.get_atomic_numbers(), dtype=torch.long, device=device)
    pos0 = torch.as_tensor(at0.get_positions(), dtype=torch.float32, device=device)

    # 基准能量（点预测）
    with torch.no_grad():
        E0_tensor = model(Z0, pos0, edge_index=None)
    E0 = to_float(E0_tensor)  # 纯 float

    N = len(at0)
    sigma_small = 0.01  # 用最小扰动做“线性”近似
    repeats_K = max(64, args.repeats)  # 稍多一点更稳

    Ki = np.zeros(N, dtype=np.float64)

    for i in trange(N, desc="Per-atom stiffness K_i", dynamic_ncols=True):
        # 只扰动第 i 个原子
        all_preds = []
        all_norms = []
        # 按 batch 组织
        per_batch = args.batch_size
        total = repeats_K
        done = 0
        while done < total:
            nb = min(per_batch, total - done)
            # 构造这个 batch 的扰动
            pos_batch = []
            for _ in range(nb):
                eps = torch.zeros_like(pos0)
                # 对第 i 个原子加微扰（各轴独立 N(0, sigma_small)）
                jitter = torch.normal(
                    mean=torch.zeros(3, device=device, dtype=pos0.dtype),
                    std=torch.ones(3, device=device, dtype=pos0.dtype) * sigma_small
                )
                eps[i] = jitter
                pos_batch.append(pos0 + eps)
                all_norms.append(float(torch.linalg.norm(jitter).item()))
            # 预测
            with torch.no_grad():
                # 批量前向
                Z_rep = Z0.unsqueeze(0).repeat(nb, 1)  # [B,N]
                pos_rep = torch.stack(pos_batch, dim=0)  # [B,N,3]
                yb = model(Z_rep, pos_rep, edge_index=None)  # [B]（我们的 forward 已支持批量）
            all_preds.append(yb.detach().cpu().numpy().astype(np.float64))
            done += nb

        yhat = to_float64_array(np.concatenate(all_preds, axis=0))  # [R]
        norms = np.asarray(all_norms, dtype=np.float64)  # [R]

        # 绝对能量变化
        dE = np.abs(yhat - E0)  # 全是 float64，不会再递归
        # “刚度”用 |ΔE| / ||Δr|| 的均值（非常小的范数用一个 ε 防 0）
        eps_norm = 1e-12
        Ki[i] = np.mean(dE / np.maximum(norms, eps_norm))

    # 写出 per-atom 刚度
    site_out = os.path.join(args.out_dir, "site_stiffness.csv")
    pd.DataFrame({"atom_index": np.arange(N), "K_i": Ki}).to_csv(site_out, index=False)
    print("[Saved] site_stiffness.csv  -> ", site_out)

    # ---------- 可选：快速作图 ----------
    if args.make_plots:
        try:
            import matplotlib.pyplot as plt
            # 1) |ΔE| vs RMSD_local（各 sigma 分色）
            plt.figure()
            for sigma, d in df_samples.groupby("sigma"):
                plt.scatter(d["rmsd_local"], np.abs(d["dE_pred"]), s=10, alpha=0.6, label=f"σ={sigma:.2f}")
            plt.xlabel("RMSD_local (Å)")
            plt.ylabel("|ΔÊ| (predicted)")
            plt.legend()
            plt.title(f"Sensitivity scatter (base={base_id})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "scatter_absdE_vs_rmsd.png"), dpi=180)
            plt.close()

            # 2) 箱线图：按 sigma 分组（不同 k 分子图）
            ks = sorted(df_samples["k"].unique().tolist())
            for k in ks:
                d = df_samples[df_samples["k"]==k]
                order = sorted(d["sigma"].unique())
                data = [np.abs(d[d["sigma"]==s]["dE_pred"].values) for s in order]
                plt.figure()
                plt.boxplot(data, labels=[f"{s:.2f}" for s in order], showfliers=False)
                plt.xlabel("σ (Å)")
                plt.ylabel("|ΔÊ| (predicted)")
                plt.title(f"|ΔÊ| box by σ  (k={k}, base={base_id})")
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, f"box_absdE_by_sigma_k{k}.png"), dpi=180)
                plt.close()

            print("[Saved] scatter_absdE_vs_rmsd.png, box_absdE_by_sigma_k*.png")
        except Exception as e:
            print(f"[Warn] 绘图失败（可忽略）：{e}")

    # ---------- 记录 meta ----------
    meta = dict(
        base_id=base_id,
        model_path=args.model_path,
        E0=E0,
        k_list=args.k_list,
        sigma_list=args.sigma_list,
        repeats=args.repeats,
        sigma_site=args.sigma_site,
        dmin=args.dmin,
        seed=args.seed
    )
    with open(os.path.join(args.out_dir, "sensitivity_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[Done] Task 3 sensitivity analysis finished.")

if __name__ == "__main__":
    main()
