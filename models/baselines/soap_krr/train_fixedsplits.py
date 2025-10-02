# -*- coding: utf-8 -*-
"""
train_fixedsplits.py  — SOAP + KRR（线性核）基线（固定折）
更新点：
1) 特征管道中加入 L2 归一化（Normalizer(norm="l2")）
2) 用 TransformedTargetRegressor 对 y 做训练期中心化（预测时还原），等价于补了 intercept

用法保持不变，例如：
PYTHONNOUSERSITE=1 "/d/.../python.exe" train_fixedsplits.py \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --summary_json "D:\DataComp\DataCompetition\data\preproc_out\summary.json" \
  --cv_folds_json "D:\DataComp\DataCompetition\data\preproc_out\cv_folds_fixed.json" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_root "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --source_mode dir --id_mode filename \
  --rcut 5.0 --nmax 8 --lmax 6 --sigma 0.4 --pooling sum \
  --alpha_grid 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 \
  --exclude_outliers \
  --out_dir "D:\DataComp\DataCompetition\runs\soap_krr_run2_centered_l2"
"""
from __future__ import annotations
import os, json, argparse, warnings, io, sys, glob
# --- 强制 UTF-8 控制台，避免 Windows 编码报错 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from tqdm import tqdm

import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error

# 本地模块
from dataset import load_xyz_from_file  # 单帧/多帧 xyz 读取
from features import build_soap_descriptor, structure_soap_vector
from plots import plot_energy_hist

def parse_args():
    ap = argparse.ArgumentParser("Train SOAP+KRR (linear) with fixed CV splits; y-centering + L2 feature normalization")
    ap.add_argument("--energies_csv", type=str, required=True)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--cv_folds_json", type=str, required=True)
    ap.add_argument("--test_ids_json", type=str, required=True)
    ap.add_argument("--xyz_root", type=str, required=True)
    ap.add_argument("--source_mode", type=str, choices=["file", "dir"], default="dir")
    ap.add_argument("--id_mode", type=str, choices=["file_block", "filename"], default=None)
    ap.add_argument("--index_base", type=int, choices=[0, 1], default=0)

    # SOAP 超参
    ap.add_argument("--rcut", type=float, default=5.0)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--pooling", type=str, choices=["sum", "mean"], default="sum")

    # KRR 正则网格
    ap.add_argument("--alpha_grid", type=float, nargs="+",
                    default=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    ap.add_argument("--exclude_outliers", action="store_true", default=False)
    # --- NEW: 异常样本降权 ---
    ap.add_argument("--use_outlier_weights", action="store_true", default=False,
                    help="不丢弃异常样本，而是给 is_outlier==1 的样本降低权重")
    ap.add_argument("--outlier_weight", type=float, default=0.4,
                    help="异常样本权重，典型 0.3~0.5")
    ap.add_argument("--outlier_col", type=str, default="is_outlier",
                    help="台账中异常列名（默认 is_outlier）")

    ap.add_argument("--out_dir", type=str, default="./runs/soap_krr_run_centered_l2")
    return ap.parse_args()

# ========== 工具函数 ==========
def _build_id_atoms_mapping_file(xyz_file: str, id_base: int = 0):
    """多帧 .xyz：id 形如 '文件名#帧号'"""
    atoms_list, _ = load_xyz_from_file(xyz_file)
    base = os.path.splitext(os.path.basename(xyz_file))[0]
    return {f"{base}#{k}": at for k, at in enumerate(atoms_list, start=id_base)}

def _build_id_atoms_mapping_dir(xyz_dir: str, pattern: str = "*.xyz"):
    """目录：一文件一结构：id=文件名去扩展名"""
    mapping = {}
    files = sorted(glob.glob(os.path.join(xyz_dir, pattern)))
    for fp in files:
        a_list, _ = load_xyz_from_file(fp)
        if len(a_list) != 1:
            raise ValueError(f"期望每个文件 1 帧，实际 {len(a_list)} in {fp}")
        stem = os.path.splitext(os.path.basename(fp))[0]
        mapping[stem] = a_list[0]
    return mapping

def make_estimator(alpha: float) -> TransformedTargetRegressor:
    """构造带 y-中心化 + L2 归一化的 KRR（线性核）流水线"""
    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),  # 标准化
        ("norm",   Normalizer(norm="l2")),                          # L2 归一化（余弦风格）
        ("krr",    KernelRidge(kernel="linear", alpha=float(alpha)))
    ])
    # 对 y 做中心化（训练时减均值，预测时加回）：等价补 intercept
    return TransformedTargetRegressor(
        regressor=base,
        transformer=StandardScaler(with_mean=True, with_std=False)
    )

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取能量台账
    df = pd.read_csv(args.energies_csv).astype({"id": str})
    if "DeltaE" not in df.columns:
        raise SystemExit("energies_csv 必须包含列 DeltaE。")

    if args.exclude_outliers and "is_outlier" in df.columns:
        pre = len(df)
        df = df[df["is_outlier"] == 0].copy()
        print(f"[Info] 排除了 {pre - len(df)} 个异常样本（is_outlier==1）。")

    # 固定折 & 测试集
    with open(args.cv_folds_json, "r", encoding="utf-8") as f:
        cv_spec = json.load(f)
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_spec = json.load(f)
    folds = cv_spec.get("folds", [])
    test_ids = set(map(str, test_spec.get("test_ids", [])))

    # 几何映射
    id_mode = args.id_mode or ("file_block" if args.source_mode == "file" else "filename")
    if args.source_mode == "file":
        id2atoms = _build_id_atoms_mapping_file(args.xyz_root, id_base=args.index_base)
    else:
        id2atoms = _build_id_atoms_mapping_dir(args.xyz_root)

    # 对齐：只保留有几何的样本
    ids_geom = set(id2atoms.keys())
    keep = [iid for iid in df["id"].tolist() if iid in ids_geom]
    if len(keep) < len(df):
        df = df[df["id"].isin(keep)].copy()
        print(f"[Info] 几何对齐后剩余 {len(df)} 条。")

    # 训练全集/测试集划分
    test_ids = [i for i in test_ids if i in df["id"].values]
    train_all_ids = [i for i in df["id"].values if i not in test_ids]
    print(f"[Info] train_all={len(train_all_ids)}, test={len(test_ids)}")

    # 计算全集 SOAP 向量（缓存）
    print("[Info] 计算结构级 SOAP 向量...")
    soap = build_soap_descriptor(
        rcut=args.rcut, nmax=args.nmax, lmax=args.lmax, sigma=args.sigma, species=("Au",)
    )
    df = df.set_index("id", drop=False)
    # --- NEW: 构建全量样本权重（默认全 1），若启用降权则对异常样本降权 ---
    if args.use_outlier_weights and args.outlier_col in df.columns:
        w_all = np.ones(len(df), dtype=float)
        mask_out = (df[args.outlier_col].values.astype(int) == 1)
        w_all[mask_out] = float(args.outlier_weight)
        if args.exclude_outliers:
            print("[Warn] 同时指定了 --exclude_outliers 与 --use_outlier_weights，"
                  "将以 '降权' 为准，不再丢弃异常样本。")
    else:
        w_all = np.ones(len(df), dtype=float)

    id_order = df.index.tolist()
    X = np.stack([structure_soap_vector(soap, id2atoms[iid], pooling=args.pooling)
                  for iid in tqdm(id_order)], axis=0)
    y = df["DeltaE"].values.astype(float)
    np.save(os.path.join(args.out_dir, "X_cache.npy"), X)
    with open(os.path.join(args.out_dir, "id_order.txt"), "w", encoding="utf-8") as f:
        for iid in id_order:
            f.write(iid + "\n")
    id_to_idx = {iid: k for k, iid in enumerate(id_order)}

    # ========== 固定折交叉验证 ==========
    alpha_grid = list(map(float, args.alpha_grid))
    fold_mae = {a: [] for a in alpha_grid}
    used_folds = 0

    for fold_id, fold in enumerate(folds, start=1):
        tr_ids_raw = [str(i) for i in fold.get("train_ids", [])]
        va_ids_raw = [str(i) for i in fold.get("val_ids",   [])]
        tr_ids = [iid for iid in tr_ids_raw if iid in train_all_ids and iid in id_to_idx]
        va_ids = [iid for iid in va_ids_raw if iid in train_all_ids and iid in id_to_idx]
        if not tr_ids or not va_ids:
            warnings.warn(f"Fold {fold_id} 为空，跳过。")
            continue
        used_folds += 1

        tr_idx = np.array([id_to_idx[i] for i in tr_ids], dtype=int)
        va_idx = np.array([id_to_idx[i] for i in va_ids], dtype=int)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        for alpha in alpha_grid:
            pipe = make_estimator(alpha)
            fit_params = {}
            if args.use_outlier_weights and args.outlier_col in df.columns:
                # TransformedTargetRegressor 会把 **fit_params 原样传给内部的 Pipeline.fit，
                # 所以这里直接用 Pipeline 的 step 名：krr__sample_weight
                fit_params["krr__sample_weight"] = w_all[tr_idx]
            pipe.fit(X_tr, y_tr, **fit_params)

            pred = pipe.predict(X_va)
            mae = mean_absolute_error(y_va, pred)
            fold_mae[alpha].append(mae)

        print(f"[Fold {fold_id}] done. train={len(tr_ids)}, val={len(va_ids)}")

    if used_folds == 0:
        raise RuntimeError("所有折都为空？请检查 cv_folds.json 或 id 对齐。")

    alpha_scores = {a: (float(np.mean(v)), float(np.std(v)), len(v))
                    for a, v in fold_mae.items() if len(v) > 0}
    alpha_star = sorted(alpha_scores.items(), key=lambda kv: (kv[1][0], -kv[0]))[0][0]
    print(f"[Best alpha] {alpha_star:.2e}  | CV_MAE={alpha_scores[alpha_star][0]:.6f}")

    # ========== 最终训练（train_all） ==========
    tr_idx_all = np.array([id_to_idx[i] for i in train_all_ids if i in id_to_idx], dtype=int)
    final_pipe = make_estimator(alpha_star)
    fit_params_all = {}
    if args.use_outlier_weights and args.outlier_col in df.columns:
        fit_params_all["krr__sample_weight"] = w_all[tr_idx_all]
    final_pipe.fit(X[tr_idx_all], y[tr_idx_all], **fit_params_all)

    # ========== 落盘 ==========
    meta = dict(
        soap=dict(rcut=args.rcut, nmax=args.nmax, lmax=args.lmax, sigma=args.sigma, pooling=args.pooling),
        krr=dict(alpha=float(alpha_star), kernel="linear"),
        y_centering=True,
        l2_normalize=True,
        cv={str(a): {"mean_mae": mu, "std": sd, "nfolds": n} for a, (mu, sd, n) in alpha_scores.items()},
        counts=dict(n_total=int(len(df)), n_train_all=int(len(tr_idx_all)),
                    n_test=int(len(test_ids)), n_features=int(X.shape[1]), used_folds=int(used_folds)),
        notes="Feature L2-normalization + target centering via TransformedTargetRegressor.",
        use_outlier_weights=bool(args.use_outlier_weights),
        outlier_weight=float(args.outlier_weight)

    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    joblib.dump(final_pipe, os.path.join(args.out_dir, "model.joblib"))
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    plot_energy_hist(df["DeltaE"].values, os.path.join(args.out_dir, "energy_hist.png"))
    print("[Done] model saved to", os.path.join(args.out_dir, "model.joblib"))

if __name__ == "__main__":
    main()
