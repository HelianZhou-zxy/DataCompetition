# -*- coding: utf-8 -*-
"""
根据 energies_nohash.csv + test_ids_nohash.json + 几何目录，生成
5 折 StratifiedGroupKFold（按 energy_bucket 分层，按 group_id 分组）。
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def load_valid_ids(energies_csv: str, xyz_dir: str):
    df = pd.read_csv(energies_csv).astype({"id": str})
    file_ids = set(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(xyz_dir, "*.xyz")))
    keep = df["id"].isin(file_ids)
    return df[keep].copy()

def make_y_stratify(df_tr: pd.DataFrame) -> np.ndarray:
    """返回用于分层的离散标签 y。优先 energy_bucket；否则按 ΔE 分位数。"""
    if "energy_bucket" in df_tr.columns and df_tr["energy_bucket"].notna().any():
        ser = df_tr["energy_bucket"]
        if pd.api.types.is_numeric_dtype(ser):
            y = ser.astype(int).values
        else:
            # 尝试常见字符串映射
            mapping = {
                "low": 0, "lo": 0, "0": 0,
                "mid": 1, "medium": 1, "med": 1, "1": 1,
                "high": 2, "hi": 2, "2": 2
            }
            y_map = ser.astype(str).str.strip().str.lower().map(mapping)
            if y_map.isna().any():
                # 仍有未知标签：使用类别编码兜底
                y = pd.Categorical(ser.astype(str)).codes
            else:
                y = y_map.values.astype(int)
    else:
        # 没有 energy_bucket：按 ΔE 十分位分桶
        q = np.linspace(0, 1, 11)
        edges = np.unique(np.quantile(df_tr["DeltaE"].values, q))
        y = np.digitize(df_tr["DeltaE"].values, edges[1:-1], right=False)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--xyz_dir", required=True)
    ap.add_argument("--test_ids_json", required=True)
    ap.add_argument("--out_folds_json", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_valid_ids(args.energies_csv, args.xyz_dir)
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_ids = set(map(str, json.load(f)["test_ids"]))

    # 训练全集 = 去掉测试集
    df_tr = df[~df["id"].isin(test_ids)].copy()
    if df_tr.empty:
        raise SystemExit("训练集为空：请检查 test_ids_nohash.json 是否覆盖了全部样本。")

    # y（分层标签）与分组
    y = make_y_stratify(df_tr)
    if "group_id" in df_tr.columns and df_tr["group_id"].notna().any():
        groups = df_tr["group_id"].astype(str).values
    else:
        groups = df_tr["id"].astype(str).values  # 退化为普通分层

    ids = df_tr["id"].astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    folds_out = []
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros_like(y), y, groups), start=1):
        tr_ids = ids[tr_idx].tolist()
        va_ids = ids[va_idx].tolist()
        # 保险：互斥 & 去重
        tr_set, va_set = set(tr_ids), set(va_ids)
        if tr_set & va_set:
            va_ids = [i for i in va_ids if i not in tr_set]
        folds_out.append({"train_ids": tr_ids, "val_ids": va_ids})
        # 小结
        print(f"[Fold{k}] train={len(tr_ids)}  val={len(va_ids)}")

    with open(args.out_folds_json, "w", encoding="utf-8") as f:
        json.dump({"folds": folds_out}, f, indent=2, ensure_ascii=False)
    print("[Saved]", args.out_folds_json)

if __name__ == "__main__":
    main()
