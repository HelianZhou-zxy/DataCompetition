# -*- coding: utf-8 -*-
"""
eval_gnn_fixedtest.py — 在固定测试集上评测已训练的 SchNet GNN
"""
import os, json, argparse, io, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# 控制台 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- 把“项目根”放进 sys.path，保证能导入 models/... 与 train_gnn_schnet ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))                       # .../models/gnn/schnet
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir))  # .../DataCompetition
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from torch.utils.data import DataLoader

from models.baselines.soap_krr.dataset import load_xyz_from_file  # 仅触发依赖检查
from models.baselines.soap_krr.model import regression_metrics

# 复用训练里的定义（批量前向已支持）
from models.gnn.schnet.train_gnn_schnet import SchNetEnergy, Au20Dataset, collate

def parse_args():
    ap = argparse.ArgumentParser("Eval SchNet GNN on fixed test set")
    ap.add_argument("--energies_csv", required=True)
    ap.add_argument("--test_ids_json", required=True)
    ap.add_argument("--xyz_dir", required=True)
    ap.add_argument("--model_path", required=True)   # e.g., model_fold1.pt
    # 需要与训练时一致
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--n_rbf", type=int, default=64)
    ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--batch_size", type=int, default=256)  # 推理可更大
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()

def set_device(dev):
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = set_device(args.device)

    df = pd.read_csv(args.energies_csv).astype({"id": str})
    with open(args.test_ids_json, "r", encoding="utf-8") as f:
        test_ids = [str(i) for i in json.load(f)["test_ids"]]

    # Dataset / Loader
    ds_te = Au20Dataset(test_ids, df, args.xyz_dir, outlier_weights=None)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    # Model
    model = SchNetEnergy(hidden_dim=args.hidden_dim, n_blocks=args.n_blocks,
                         n_rbf=args.n_rbf, rcut=args.rcut).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ids_all, y_true, y_pred = [], [], []
    with torch.no_grad():
        for ids, Z, pos, y, w in tqdm(dl_te, dynamic_ncols=True, desc="Eval"):
            Z = Z.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                yhat = model(Z, pos, edge_index=None)   # ★ 批量前向
            ids_all.extend(list(ids))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(yhat.detach().cpu().numpy().tolist())

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    metrics = regression_metrics(y_true, y_pred)
    print("[Metrics]", metrics)

    out_pred = os.path.join(args.out_dir, "test_predictions_gnn.csv")
    out_met  = os.path.join(args.out_dir, "metrics_gnn.json")
    pd.DataFrame({"id": ids_all, "y_true": y_true, "y_pred": y_pred}).to_csv(out_pred, index=False)
    with open(out_met, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[Saved]", out_met)
    print("[Saved]", out_pred)

if __name__ == "__main__":
    main()
