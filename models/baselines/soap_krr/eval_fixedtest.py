# -*- coding: utf-8 -*-
"""
eval_fixedtest.py — 使用固定测试集 (test_ids.json) 对已训练的 SOAP+KRR 模型做最终评测，
输出整体与分位 MAE、parity 图，以及逐样本预测残差。

输入：
  - model.joblib, meta.json
  - energies.csv (提供 DeltaE 与 energy_bucket)
  - test_ids.json
  - xyz 源（同训练使用的源与 id 解析方式）

输出：
  - test_predictions.csv (id, y_true, y_pred, residual, energy_bucket)
  - metrics.json
  - parity.png, binned_mae.png
"""
from __future__ import annotations
import os, json, argparse, joblib, warnings
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import load_xyz_from_file, load_xyz_from_dir
from features import build_soap_descriptor, structure_soap_vector
from model import regression_metrics
from plots import parity_plot, binned_mae
import sys, os, io
sys.path.insert(0, os.path.dirname(__file__))  # 让同目录的 dataset/features 等可被找到
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
def parse_args():
    ap = argparse.ArgumentParser("Evaluate trained SOAP+KRR on fixed test set")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--meta_path", type=str, required=True)
    ap.add_argument("--energies_csv", type=str, required=True)
    ap.add_argument("--test_ids_json", type=str, required=True)

    ap.add_argument("--xyz_root", type=str, required=True)
    ap.add_argument("--source_mode", type=str, choices=["file","dir"], default="file")
    ap.add_argument("--id_mode", type=str, choices=["file_block","filename"], default=None)
    ap.add_argument("--index_base", type=int, choices=[0,1], default=0)

    ap.add_argument("--out_dir", type=str, default="./runs/soap_krr_run_fixed")
    return ap.parse_args()

def _build_id_atoms_mapping_file(xyz_file: str, id_base: int = 0) -> Dict[str, "Atoms"]:
    import os
    from dataset import load_xyz_from_file
    atoms_list, _ = load_xyz_from_file(xyz_file)
    base = os.path.splitext(os.path.basename(xyz_file))[0]
    mapping = {}
    for k, at in enumerate(atoms_list, start=id_base):
        mapping[f"{base}#{k}"] = at
    return mapping

def _build_id_atoms_mapping_dir(xyz_dir: str, pattern: str = "*.xyz") -> Dict[str, "Atoms"]:
    import glob, os
    mapping = {}
    files = sorted(glob.glob(os.path.join(xyz_dir, pattern)))
    for fp in files:
        atoms_list, _ = load_xyz_from_file(fp)
        if len(atoms_list) != 1:
            raise ValueError(f"Expect 1-frame xyz per file, got {len(atoms_list)} in {fp}")
        stem = os.path.splitext(os.path.basename(fp))[0]
        mapping[stem] = atoms_list[0]
    return mapping

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载模型与元信息（含 SOAP 超参）
    model = joblib.load(args.model_path)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    soap_hp = meta["soap"]
    pooling = soap_hp.get("pooling", "sum")

    # 读取台账与测试集 ID
    df = pd.read_csv(args.energies_csv).set_index("id", drop=False)
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_spec = json.load(f)
    test_ids = [str(i) for i in test_spec.get("test_ids", [])]

    # 构建几何映射
    id_mode = args.id_mode or ("file_block" if args.source_mode == "file" else "filename")
    if args.source_mode == "file":
        id2atoms = _build_id_atoms_mapping_file(args.xyz_root, id_base=args.index_base)
    else:
        id2atoms = _build_id_atoms_mapping_dir(args.xyz_root)

    # 对齐与过滤：仅保留 energies.csv 与 几何源都存在的 test_ids
    test_ids = [iid for iid in test_ids if (iid in df.index and iid in id2atoms)]
    if not test_ids:
        raise RuntimeError("test_ids 为空或不在 energies/几何源中，请检查输入。")

    # 计算测试集 SOAP 向量
    soap = build_soap_descriptor(
        rcut=soap_hp["rcut"], nmax=soap_hp["nmax"], lmax=soap_hp["lmax"],
        sigma=soap_hp["sigma"], species=("Au",)
    )
    X_te, y_te, buckets = [], [], []
    for iid in tqdm(test_ids, desc="SOAP(test)"):
        v = structure_soap_vector(soap, id2atoms[iid], pooling=pooling)
        X_te.append(v)
        y_te.append(float(df.loc[iid, "DeltaE"]))
        buckets.append(int(df.loc[iid, "energy_bucket"]) if "energy_bucket" in df.columns else -1)
    X_te = np.stack(X_te, axis=0)
    y_te = np.asarray(y_te, dtype=float)

    # 预测与评估
    y_pred = model.predict(X_te)
    metrics = regression_metrics(y_te, y_pred)

    # 输出预测与图表
    out_pred_path = os.path.join(args.out_dir, "test_predictions.csv")
    with open(out_pred_path, "w", encoding="utf-8") as f:
        f.write("id,y_true,y_pred,residual,energy_bucket\n")
        for iid, yt, yp, eb in zip(test_ids, y_te, y_pred, buckets):
            f.write(f"{iid},{yt:.10f},{yp:.10f},{(yp-yt):.10f},{eb}\n")

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    parity_plot(y_te, y_pred, os.path.join(args.out_dir, "parity.png"))
    binned_mae(y_te, y_pred, os.path.join(args.out_dir, "binned_mae.png"))

    print("[Test metrics]", metrics)
    print("[Saved]", out_pred_path)

if __name__ == "__main__":
    main()
