# -*- coding: utf-8 -*-
"""
viz_report.py — 生成四类可视化：
1) Parity（点预测，双模型同图）
2) Residual vs ΔE（各自一图，按能量桶上色）
3) CP 区间带（按 ΔE 排序，alpha 指定；双模型并排）
4) 各桶区间宽度箱线图（global/Mondrian 一样的用法；q_by_bucket -> per-sample 宽度）

依赖：matplotlib, pandas, numpy
输入：每个 run 目录应包含：
  - test_predictions_gnn.csv（id, y_true, y_pred）
  - metrics_gnn.json（可选，用来在图上标注 MAE/RMSE/R²）
  - metrics_cp.json（包含 cp.{alpha}.q_by_bucket 与 PICP/MPIW）
以及全局的 energies_csv（含 id, DeltaE, energy_bucket）
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKET_ORDER = ["low", "mid", "high"]

def load_run(run_dir):
    # 允许两种文件名（稳一点）
    cand = ["test_predictions_gnn.csv", "test_predictions.csv"]
    pred_path = None
    for c in cand:
        p = os.path.join(run_dir, c)
        if os.path.exists(p):
            pred_path = p
            break
    if pred_path is None:
        raise FileNotFoundError(f"未找到预测文件：{cand} 任一在 {run_dir}")

    df_pred = pd.read_csv(pred_path).astype({"id": str})
    met_gnn = {}
    mg = os.path.join(run_dir, "metrics_gnn.json")
    if os.path.exists(mg):
        with open(mg, "r", encoding="utf-8") as f:
            met_gnn = json.load(f)
    met_cp = {}
    mc = os.path.join(run_dir, "metrics_cp.json")
    if os.path.exists(mc):
        with open(mc, "r", encoding="utf-8") as f:
            met_cp = json.load(f)
    return df_pred, met_gnn, met_cp

def attach_meta(df_pred, energies_csv):
    df_e = pd.read_csv(energies_csv).astype({"id": str})
    cols_need = ["id", "DeltaE", "energy_bucket"]
    miss = [c for c in cols_need if c not in df_e.columns]
    if miss:
        raise ValueError(f"energies_csv 缺少列：{miss}")
    df = df_pred.merge(df_e[cols_need], on="id", how="left")
    # 桶设为有序类别，绘图更好看
    df["energy_bucket"] = pd.Categorical(df["energy_bucket"], categories=BUCKET_ORDER, ordered=True)
    return df

def per_sample_q(df, met_cp, alpha):
    """
    从 metrics_cp.json 取 cp[alpha].q_by_bucket，映射到每个样本；
    若不存在 q_by_bucket，尝试取全局 q；若都没有，返回 None。
    """
    cp = met_cp.get("cp", {})
    key = f"{alpha:.1f}".rstrip("0").rstrip(".")  # 0.1 -> "0.1"
    entry = cp.get(key, None)
    if entry is None:
        # 兼容数字键
        entry = cp.get(alpha, None)
    if entry is None:
        return None

    q_map = entry.get("q_by_bucket", None)
    if q_map is None:
        q = entry.get("q", None)
        if q is None:
            return None
        # 全局同一个 q
        return np.full(len(df), float(q), dtype=float)

    # bucket 专属 q（Mondrian/global 都支持这种写法）
    q_ser = df["energy_bucket"].map(lambda b: float(q_map.get(str(b), np.nan)))
    return q_ser.to_numpy()

def parity_plot(dfA, nameA, dfB, nameB, out_png, metA=None, metB=None):
    plt.figure(figsize=(6,6))
    xy = np.concatenate([dfA[["y_true"]].to_numpy(), dfB[["y_true"]].to_numpy()])
    y_min, y_max = np.min(xy), np.max(xy)
    pad = 0.05*(y_max - y_min + 1e-6)
    lo, hi = y_min - pad, y_max + pad

    plt.plot([lo, hi],[lo, hi], linestyle="--", linewidth=1)
    plt.scatter(dfA["y_true"], dfA["y_pred"], s=12, alpha=0.6, label=f"{nameA}")
    plt.scatter(dfB["y_true"], dfB["y_pred"], s=12, alpha=0.6, label=f"{nameB}")
    # 在角落标注指标
    if metA:
        plt.text(0.02, 0.98, f"{nameA}\nMAE={metA.get('MAE','-'):.3f}\nRMSE={metA.get('RMSE','-'):.3f}\nR²={metA.get('R2','-'):.3f}",
                 transform=plt.gca().transAxes, va="top")
    if metB:
        plt.text(0.98, 0.98, f"{nameB}\nMAE={metB.get('MAE','-'):.3f}\nRMSE={metB.get('RMSE','-'):.3f}\nR²={metB.get('R2','-'):.3f}",
                 transform=plt.gca().transAxes, va="top", ha="right")
    plt.xlabel("True ΔE")
    plt.ylabel("Predicted ΔE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def residual_vs_deltaE(df, title, out_png):
    # 残差按桶上色
    colors = {"low":None, "mid":None, "high":None}  # 让 matplotlib 自配色
    plt.figure(figsize=(7,4))
    for b in BUCKET_ORDER:
        d = df[df["energy_bucket"]==b]
        if len(d)==0: continue
        plt.scatter(d["DeltaE"], d["y_pred"]-d["y_true"], s=14, alpha=0.7, label=b)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("ΔE (true)")
    plt.ylabel("Residual (pred - true)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def cp_bands(df, qs, title, out_png):
    # 根据 ΔE 排序，画 y_true、y_pred 及 y_pred±q 的带子
    d = df.sort_values("DeltaE").reset_index(drop=True).copy()
    lower = d["y_pred"] - qs
    upper = d["y_pred"] + qs

    plt.figure(figsize=(8,4))
    plt.plot(d.index, d["y_true"], linewidth=1, label="True")
    plt.plot(d.index, d["y_pred"], linewidth=1, label="Pred")
    plt.fill_between(d.index, lower, upper, alpha=0.25, label="Pred ± q")
    plt.xlabel("Sorted by ΔE")
    plt.ylabel("ΔE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def cp_width_box(dfA, qA, nameA, dfB, qB, nameB, out_png):
    # 每个样本区间宽度 = 2*q；按桶分组画并列箱线图
    widths = []
    labels = []
    positions = []
    pos = 1
    for b in BUCKET_ORDER:
        wA = 2.0 * qA[dfA["energy_bucket"]==b]
        wB = 2.0 * qB[dfB["energy_bucket"]==b]
        widths.append(wA.dropna().to_numpy()); labels.append(f"{b}\n{nameA}"); positions.append(pos); pos += 1
        widths.append(wB.dropna().to_numpy()); labels.append(f"{b}\n{nameB}"); positions.append(pos); pos += 1
        pos += 0.5  # 桶之间留空

    plt.figure(figsize=(10,4))
    plt.boxplot(widths, positions=positions, widths=0.6, showfliers=False)
    plt.xticks(positions, labels, rotation=0)
    plt.ylabel("Interval Width (2q)")
    plt.title("Conformal Interval Widths by Energy Bucket")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser("Make figures for KD-only vs KD+VKD")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--runA_dir", required=True, help="例如 KD-only 的目录")
    ap.add_argument("--runA_name", default="KD-only")
    ap.add_argument("--runB_dir", required=True, help="例如 KD+VKD 的目录")
    ap.add_argument("--runB_name", default="KD+VKD")
    ap.add_argument("--alpha", type=float, default=0.1, help="选择可视化的 CP 置信水平 1-α")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载两套 run
    dfA_pred, metA, cpA = load_run(args.runA_dir)
    dfB_pred, metB, cpB = load_run(args.runB_dir)

    # 附加元数据
    dfA = attach_meta(dfA_pred, args.energies_csv)
    dfB = attach_meta(dfB_pred, args.energies_csv)
    dfA["resid"] = dfA["y_pred"] - dfA["y_true"]
    dfB["resid"] = dfB["y_pred"] - dfB["y_true"]

    # 取 CP 分位并映射每样本
    qA = per_sample_q(dfA, cpA, args.alpha)
    qB = per_sample_q(dfB, cpB, args.alpha)
    if qA is None or qB is None:
        raise RuntimeError(f"metrics_cp.json 里找不到 alpha={args.alpha} 的分位（q 或 q_by_bucket）")

    # 图1：Parity
    out1 = os.path.join(args.out_dir, f"fig1_parity.png")
    parity_plot(dfA, args.runA_name, dfB, args.runB_name, out1, metA, metB)

    # 图2：Residual vs ΔE（两张）
    out2a = os.path.join(args.out_dir, f"fig2_residual_vs_deltaE_{args.runA_name}.png")
    out2b = os.path.join(args.out_dir, f"fig2_residual_vs_deltaE_{args.runB_name}.png")
    residual_vs_deltaE(dfA, f"Residual vs ΔE — {args.runA_name}", out2a)
    residual_vs_deltaE(dfB, f"Residual vs ΔE — {args.runB_name}", out2b)

    # 图3：CP 区间带（两张）
    out3a = os.path.join(args.out_dir, f"fig3_cp_bands_alpha{args.alpha}_{args.runA_name}.png")
    out3b = os.path.join(args.out_dir, f"fig3_cp_bands_alpha{args.alpha}_{args.runB_name}.png")
    cp_bands(dfA.copy(), qA, f"CP Bands α={args.alpha} — {args.runA_name}", out3a)
    cp_bands(dfB.copy(), qB, f"CP Bands α={args.alpha} — {args.runB_name}", out3b)

    # 图4：各桶区间宽度箱线图（并列）
    out4 = os.path.join(args.out_dir, f"fig4_cp_width_box_alpha{args.alpha}.png")
    # 为了方便 dropna，用 Series 包装
    qA_ser = pd.Series(qA, index=dfA.index)
    qB_ser = pd.Series(qB, index=dfB.index)
    cp_width_box(dfA, qA_ser, args.runA_name, dfB, qB_ser, args.runB_name, out4)

    print("[Saved]", out1)
    print("[Saved]", out2a)
    print("[Saved]", out2b)
    print("[Saved]", out3a)
    print("[Saved]", out3b)
    print("[Saved]", out4)

if __name__ == "__main__":
    main()
