# -*- coding: utf-8 -*-
"""
eval_ensemble.py — 固定测试集上的多模型简单平均集成
用法：
PYTHONNOUSERSITE=1 ".../python.exe" eval_ensemble.py \
  --runs "D:\...\runs\soap_krr_run2_centered_l2" \
         "D:\...\runs\soap_krr_run3_rcut6_n10_l8_s0p5_w04" \
         "D:\...\runs\soap_krr_run4_rcut5p5_n8_l8_s0p3" \
  --energies_csv "D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv" \
  --test_ids_json "D:\DataComp\DataCompetition\data\preproc_out\test_ids_nohash.json" \
  --xyz_root "D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000" \
  --source_mode dir --id_mode filename \
  --out_dir "D:\DataComp\DataCompetition\runs\soap_krr_ENSEMBLE"
"""
from __future__ import annotations
import os, io, sys, json, argparse, glob
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from dataset import load_xyz_from_file
from features import build_soap_descriptor, structure_soap_vector
from model import regression_metrics
from plots import parity_plot, binned_mae

def _build_id_atoms_mapping_dir(xyz_dir: str, pattern: str="*.xyz"):
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
    ap = argparse.ArgumentParser("Ensemble evaluation on fixed test IDs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="多个训练 run 目录，每个目录需包含 model.joblib 与 meta.json")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--test_ids_json", required=True)
    ap.add_argument("--xyz_root", required=True)
    ap.add_argument("--source_mode", choices=["dir","file"], default="dir")
    ap.add_argument("--id_mode", choices=["filename","file_block"], default="filename")
    ap.add_argument("--index_base", type=int, choices=[0,1], default=0)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 台账 & 测试ID
    df = pd.read_csv(args.energies_csv).astype({"id": str})
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_ids = [str(i) for i in json.load(f)["test_ids"]]
    df = df.set_index("id", drop=False)
    y_test = df.loc[test_ids, "DeltaE"].astype(float).values

    # 几何
    if args.source_mode == "dir":
        id2atoms = _build_id_atoms_mapping_dir(args.xyz_root)
    else:
        a_list, _ = load_xyz_from_file(args.xyz_root)
        base = os.path.splitext(os.path.basename(args.xyz_root))[0]
        id2atoms = {f"{base}#{i}": a for i, a in enumerate(a_list, start=args.index_base)}
    for iid in test_ids:
        if iid not in id2atoms:
            raise SystemExit(f"几何缺失: {iid}")

    # 逐模型预测
    preds = []
    for run in args.runs:
        model_path = os.path.join(run, "model.joblib")
        meta_path  = os.path.join(run, "meta.json")
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise SystemExit(f"[{run}] 缺少 model.joblib/meta.json")
        pipe = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        soap_hp = meta.get("soap", {})
        soap = build_soap_descriptor(
            rcut=float(soap_hp.get("rcut", 5.0)),
            nmax=int(soap_hp.get("nmax", 8)),
            lmax=int(soap_hp.get("lmax", 6)),
            sigma=float(soap_hp.get("sigma", 0.4)),
            species=("Au",)
        )
        pooling = soap_hp.get("pooling", "sum")
        X = np.stack(
            [structure_soap_vector(soap, id2atoms[iid], pooling=pooling) for iid in tqdm(test_ids, desc=f"SOAP({os.path.basename(run)})")],
            axis=0
        )
        y_pred = pipe.predict(X)
        preds.append(y_pred)
        m = regression_metrics(y_test, y_pred)
        print(f"[{os.path.basename(run)}] MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")

    # 简单平均（也可以换中位数）
    Y = np.vstack(preds)  # [n_models, n_test]
    y_ens = Y.mean(axis=0)
    metrics = regression_metrics(y_test, y_ens)
    print("[Ensemble]", metrics)

    # 输出
    out_pred = os.path.join(args.out_dir, "test_predictions_ensemble.csv")
    out_met  = os.path.join(args.out_dir, "metrics_ensemble.json")
    pd.DataFrame({"id": test_ids, "y_true": y_test, "y_pred": y_ens}).to_csv(out_pred, index=False, encoding="utf-8")
    with open(out_met, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 图
    parity_plot(y_test, y_ens, os.path.join(args.out_dir, "parity_ensemble.png"))
    binned_mae(y_test, y_ens, os.path.join(args.out_dir, "binned_mae_ensemble.png"))
    print("[Saved]", out_met)
    print("[Saved]", out_pred)
    print("[Done]")

if __name__ == "__main__":
    main()
