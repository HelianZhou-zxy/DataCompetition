# -*- coding: utf-8 -*-
"""
eval_fixedtest_robust.py
- 固定测试集评测（DeltaE），对齐更健壮 + 详细诊断输出
- 读取 meta.json 的 SOAP 超参，重算 test 的 SOAP 向量
- 输出 metrics.json / test_predictions.csv / parity.png / binned_mae.png
"""
from __future__ import annotations
import os, io, sys, json, argparse, joblib, warnings
sys.path.insert(0, os.path.dirname(__file__))  # 本地导入
# --- 强制 UTF-8 控制台 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import load_xyz_from_file
from features import build_soap_descriptor, structure_soap_vector
from model import regression_metrics
from plots import parity_plot, binned_mae

def _build_id_atoms_mapping_dir(xyz_dir: str, pattern: str="*.xyz"):
    import glob, os
    mapping = {}
    files = sorted(glob.glob(os.path.join(xyz_dir, pattern)))
    for fp in files:
        a_list, _ = load_xyz_from_file(fp)
        if len(a_list) != 1:
            raise ValueError(f"Expect 1-frame xyz per file, got {len(a_list)} in {fp}")
        stem = os.path.splitext(os.path.basename(fp))[0]
        mapping[stem] = a_list[0]
    return mapping

def parse_args():
    ap = argparse.ArgumentParser("Robust evaluation on fixed test IDs")
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--meta_path",  required=True, type=str)  # 读取 SOAP 超参
    ap.add_argument("--energies_csv", required=True, type=str)
    ap.add_argument("--test_ids_json", required=True, type=str)
    ap.add_argument("--xyz_root", required=True, type=str)
    ap.add_argument("--source_mode", choices=["dir","file"], default="dir")
    ap.add_argument("--id_mode", choices=["filename","file_block"], default="filename")
    ap.add_argument("--index_base", type=int, choices=[0,1], default=0)  # file_block 才有用
    ap.add_argument("--exclude_outliers", action="store_true", default=False)
    ap.add_argument("--out_dir", required=True, type=str)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 读取 model + meta（拿 SOAP 超参）
    pipe = joblib.load(args.model_path)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    soap_hp = meta.get("soap", {})
    rcut  = float(soap_hp.get("rcut", 5.0))
    nmax  = int(soap_hp.get("nmax", 8))
    lmax  = int(soap_hp.get("lmax", 6))
    sigma = float(soap_hp.get("sigma", 0.4))
    pooling = soap_hp.get("pooling", "sum")

    print(f"[Meta] SOAP(r={rcut}, n={nmax}, l={lmax}, σ={sigma}, pool={pooling})")

    # 2) energies + test_ids
    df = pd.read_csv(args.energies_csv).astype({"id": str})
    if "DeltaE" not in df.columns:
        raise SystemExit("energies_csv 必须包含列 DeltaE。")
    if args.exclude_outliers and "is_outlier" in df.columns:
        pre = len(df); df = df[df["is_outlier"] == 0].copy()
        print(f"[Info] 排除了 {pre - len(df)} 个异常样本（is_outlier==1）。")

    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_spec = json.load(f)
    raw_test_ids = [str(i) for i in test_spec.get("test_ids", [])]
    print(f"[Diag] test_ids.json 原始数量 = {len(raw_test_ids)} （示例: {raw_test_ids[:8]}）")

    # 3) 几何映射
    if args.source_mode == "dir":
        id2atoms = _build_id_atoms_mapping_dir(args.xyz_root)
    else:
        # 多帧文件模式（本项目不用；保留占位实现）
        from dataset import load_xyz_from_file
        a_list, _ = load_xyz_from_file(args.xyz_root)
        base = os.path.splitext(os.path.basename(args.xyz_root))[0]
        id2atoms = {f"{base}#{i}": a for i, a in enumerate(a_list, start=args.index_base)}

    ids_geom = set(id2atoms.keys())
    ids_ener = set(df["id"].astype(str).tolist())

    # 4) 逐步过滤 + 诊断
    test_in_ener = [i for i in raw_test_ids if i in ids_ener]
    miss_ener = [i for i in raw_test_ids if i not in ids_ener]
    print(f"[Diag] 在 energies_csv 命中的 test_ids = {len(test_in_ener)}，缺失 = {len(miss_ener)}")
    if miss_ener:
        print(f"       例子（最多10个）：{miss_ener[:10]}")

    test_in_geom = [i for i in raw_test_ids if i in ids_geom]
    miss_geom = [i for i in raw_test_ids if i not in ids_geom]
    print(f"[Diag] 在几何目录命中的 test_ids = {len(test_in_geom)}，缺失 = {len(miss_geom)}")
    if miss_geom:
        print(f"       例子（最多10个）：{miss_geom[:10]}")

    test_ids = [i for i in raw_test_ids if i in ids_ener and i in ids_geom]
    print(f"[Diag] energies ∩ 几何 ∩ test = {len(test_ids)}")
    if len(test_ids) == 0:
        raise RuntimeError("test_ids 交集为空，请核对路径/ID 命名。")

    # 5) 构建 X_test / y_test
    df = df.set_index("id", drop=False)
    soap = build_soap_descriptor(rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, species=("Au",))
    X_list, y_list = [], []
    for iid in tqdm(test_ids, desc="SOAP(test)"):
        at = id2atoms[iid]
        X_list.append(structure_soap_vector(soap, at, pooling=pooling))
        y_list.append(float(df.loc[iid, "DeltaE"]))
    X_test = np.stack(X_list, axis=0)
    y_test = np.asarray(y_list, dtype=float)

    # 6) 预测 + 指标
    y_pred = pipe.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    print("[Metrics]", metrics)

    # 7) 输出
    out_pred = os.path.join(args.out_dir, "test_predictions.csv")
    out_met  = os.path.join(args.out_dir, "metrics.json")
    pd.DataFrame({"id": test_ids, "y_true": y_test, "y_pred": y_pred,
                  "residual": y_pred - y_test}).to_csv(out_pred, index=False, encoding="utf-8")
    with open(out_met, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    parity_plot(y_test, y_pred, os.path.join(args.out_dir, "parity.png"))
    binned_mae(y_test, y_pred, os.path.join(args.out_dir, "binned_mae.png"))

    print("[Saved]", out_met)
    print("[Saved]", out_pred)
    print("[Done]")

if __name__ == "__main__":
    main()
