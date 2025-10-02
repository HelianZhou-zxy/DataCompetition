# -*- coding: utf-8 -*-
"""
eval_gnn_conformal.py — 用验证折残差做 Split/Mondrian CP，并在固定测试集上评 coverage
"""
import os, json, argparse, sys
import numpy as np, pandas as pd
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from torch.utils.data import DataLoader
from models.gnn.schnet.train_gnn_schnet import SchNetEnergy, Au20Dataset, collate
from models.baselines.soap_krr.model import regression_metrics

def parse_args():
    ap = argparse.ArgumentParser("Conformal Prediction for SchNet GNN")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--cv_folds_json", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--test_ids_json", required=True)
    ap.add_argument("--xyz_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--n_rbf", type=int, default=64)
    ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05,0.1,0.2])
    ap.add_argument("--mondrian", action="store_true", default=False, help="按 energy_bucket 分桶做 CP")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()

def set_device(dev):
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

def predict(model, dl, device):
    ids_all, y_true, y_pred = [], [], []
    model.eval()
    with torch.no_grad():
        for ids, Z, pos, y, w in tqdm(dl, desc="Predict", dynamic_ncols=True):
            Z = Z.to(device); pos = pos.to(device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                yhat = model(Z, pos, edge_index=None)
            ids_all += list(ids)
            y_true += y.numpy().tolist()
            y_pred += yhat.detach().cpu().numpy().tolist()
    return np.array(y_true), np.array(y_pred), ids_all

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = set_device(args.device)

    df = pd.read_csv(args.energies_csv).astype({"id": str})
    with open(args.cv_folds_json, "r", encoding="utf-8") as f:
        folds = json.load(f)["folds"]
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_ids = [str(i) for i in json.load(f)["test_ids"]]

    val_ids = [str(i) for i in folds[args.fold-1]["val_ids"]]

    ds_va = Au20Dataset(val_ids,  df, args.xyz_dir, outlier_weights=None)
    ds_te = Au20Dataset(test_ids, df, args.xyz_dir, outlier_weights=None)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = SchNetEnergy(args.hidden_dim, args.n_blocks, args.n_rbf, args.rcut).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    yv, pv, ids_v = predict(model, dl_va, device)
    yt, pt, ids_t = predict(model, dl_te, device)

    base_metrics = regression_metrics(yt, pt)
    print("[Point Metrics]", base_metrics)

    # --- CP: 全局 or 分桶 ---
    results = {"point": base_metrics, "cp": {}}
    if args.mondrian:
        buckets_v = df.set_index("id").loc[ids_v, "energy_bucket"].tolist()
        buckets_t = df.set_index("id").loc[ids_t, "energy_bucket"].tolist()

    for alpha in args.alphas:
        if not args.mondrian:
            r = np.abs(pv - yv)
            # 保守分位数
            n = len(r)
            q = np.quantile(np.sort(r), min(1.0, np.ceil((n+1)*(1-alpha))/n))
            lo = pt - q; hi = pt + q
            picp = float(((yt >= lo) & (yt <= hi)).mean())
            mpiw = float((hi - lo).mean())
            results["cp"][str(alpha)] = {"q": float(q), "PICP": picp, "MPIW": mpiw}
            out_csv = pd.DataFrame({"id": ids_t, "y_true": yt, "y_pred": pt, "lo": lo, "hi": hi})
            out_csv.to_csv(os.path.join(args.out_dir, f"cp_alpha{alpha:.2f}.csv"), index=False)
        else:
            # Mondrian（按桶）
            q_map = {}
            dfv = pd.DataFrame({"id": ids_v, "bucket": buckets_v, "abs_err": np.abs(pv - yv)})
            for b, sub in dfv.groupby("bucket"):
                r = np.sort(sub["abs_err"].values)
                n = len(r)
                q_map[b] = float(np.quantile(r, min(1.0, np.ceil((n+1)*(1-alpha))/n))) if n>0 else 0.0
            # 应用到测试
            dft = pd.DataFrame({"id": ids_t, "bucket": buckets_t, "y_true": yt, "y_pred": pt})
            lo = []; hi = []
            for _, row in dft.iterrows():
                q = q_map.get(row["bucket"], 0.0)
                lo.append(row["y_pred"] - q)
                hi.append(row["y_pred"] + q)
            lo = np.array(lo); hi = np.array(hi)
            picp = float(((yt >= lo) & (yt <= hi)).mean())
            mpiw = float((hi - lo).mean())
            results["cp"][str(alpha)] = {"q_by_bucket": q_map, "PICP": picp, "MPIW": mpiw}
            out_csv = pd.DataFrame({"id": ids_t, "bucket": buckets_t, "y_true": yt, "y_pred": pt, "lo": lo, "hi": hi})
            out_csv.to_csv(os.path.join(args.out_dir, f"cp_mondrian_alpha{alpha:.2f}.csv"), index=False)

    with open(os.path.join(args.out_dir, "metrics_cp.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[Saved]", os.path.join(args.out_dir, "metrics_cp.json"))

if __name__ == "__main__":
    main()
