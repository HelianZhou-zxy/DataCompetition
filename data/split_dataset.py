"""
split_dataset.py
----------------
End-to-end preprocessing & splitting for the Au20 competition dataset.

What it does:
1) Load XYZ structures & energies (directory or single multi-xyz).
2) Compute ΔE = E - Emin and summary statistics.
3) Build geometry fingerprints (sorted pairwise distances) and group near-duplicates via DBSCAN.
4) Mark high-energy outliers (z>3) for later robustness checks.
5) Create energy quantile buckets (low/mid/high) for stratified evaluation.
6) Make a FIXED test split (~15%) at **group level** with **stratification by bucket**.
7) Build 5-fold Stratified Group K-Folds for tuning.
8) Save energies.csv, test_ids.json, cv_folds.json, summary.json.

Usage:
  # Directory of xyz files:
  python split_dataset.py --data_dir ./data/xyz --pattern "*.xyz" --out_dir ./preproc_out

  # Single multi-structure xyz:
  python split_dataset.py --xyz_file ./data/all.xyz --out_dir ./preproc_out
"""

from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict

from dataset_io import load_xyz_directory, load_xyz_file
from geometry_fingerprints import pairwise_distances_fingerprint

RANDOM_SEED = 2025

def compute_fingerprints(structs: List[dict]) -> np.ndarray:
    fps = []
    for s in structs:
        fps.append(pairwise_distances_fingerprint(s["coords"]))
    return np.vstack(fps)

def auto_dbscan_eps(X: np.ndarray, pct: float = 5.0) -> float:
    """Choose eps as the given percentile of 1-NN distances after scaling."""
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    dists, _ = nn.kneighbors(X)
    one_nn = np.sort(dists[:, 1])
    eps = float(np.percentile(one_nn, pct))
    return max(eps, 1e-6)

def group_by_dbscan(fps: np.ndarray, min_samples: int = 2, eps: float = None) -> np.ndarray:
    X = StandardScaler().fit_transform(fps)
    if eps is None:
        eps = auto_dbscan_eps(X, pct=5.0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X).labels_.copy()
    if (labels == -1).any():
        noise = np.where(labels == -1)[0]
        mx = labels.max()
        for k, idx in enumerate(noise, start=1):
            labels[idx] = mx + k
    return labels

def mark_outliers(deltaE: np.ndarray, z: float = 3.0) -> np.ndarray:
    mu = float(np.mean(deltaE))
    sigma = float(np.std(deltaE) + 1e-12)
    zscores = (deltaE - mu) / sigma
    return (zscores > z).astype(int)

def quantile_buckets(deltaE: np.ndarray, q=(0.25, 0.75)) -> List[str]:
    q1, q3 = np.quantile(deltaE, q)
    out = []
    for v in deltaE:
        if v <= q1:
            out.append("low")
        elif v <= q3:
            out.append("mid")
        else:
            out.append("high")
    return out

def stratified_group_test_split(df: pd.DataFrame, test_frac: float = 0.15, seed: int = RANDOM_SEED):
    rng = np.random.RandomState(seed)
    # majority bucket per group
    g_major = (
        df.groupby("group_id")["energy_bucket"]
          .agg(lambda s: Counter(s).most_common(1)[0][0])
          .reset_index()
    )
    g_sizes = df.groupby("group_id")["id"].count().rename("size").reset_index()
    gtable = g_major.merge(g_sizes, on="group_id", how="left")
    chosen = []
    for label, gdf in gtable.groupby("energy_bucket", sort=False):
        n = max(1, int(round(test_frac * len(gdf))))
        chosen.extend(gdf.sample(n=n, random_state=rng).group_id.tolist())
    chosen = set(chosen)
    test_ids = df[df.group_id.isin(chosen)].id.tolist()
    train_ids = df[~df.group_id.isin(chosen)].id.tolist()
    return train_ids, test_ids

def stratified_group_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED):
    rng = np.random.RandomState(seed)
    groups = df.group_id.unique().tolist()
    rng.shuffle(groups)
    # bucket counts per group
    gb = df.groupby(["group_id", "energy_bucket"]).size().unstack(fill_value=0)
    folds_hist = [defaultdict(int) for _ in range(n_splits)]
    fold_groups = [[] for _ in range(n_splits)]
    for g in groups:
        counts = gb.loc[g].to_dict()
        best_f, best_score = None, None
        for f in range(n_splits):
            tmp = folds_hist[f].copy()
            for k, v in counts.items():
                tmp[k] += int(v)
            # simple L2 imbalance score against running average
            avg = {k: (sum(ff[k] for ff in folds_hist) + counts.get(k,0)) / (f+1) for k in tmp.keys()}
            score = sum((tmp[k]-avg[k])**2 for k in tmp.keys())
            if best_score is None or score < best_score:
                best_score, best_f = score, f
        fold_groups[best_f].append(g)
        for k, v in counts.items():
            folds_hist[best_f][k] += int(v)
    folds = []
    for f in range(n_splits):
        val_g = set(fold_groups[f])
        val_ids = df[df.group_id.isin(val_g)].id.tolist()
        train_ids = df[~df.group_id.isin(val_g)].id.tolist()
        folds.append({"train_ids": train_ids, "val_ids": val_ids})
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help="Directory of .xyz files")
    ap.add_argument("--xyz_file", type=str, default=None, help="Single multi-structure .xyz file")
    ap.add_argument("--pattern", type=str, default="*.xyz")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dbscan_min_samples", type=int, default=2)
    ap.add_argument("--dbscan_eps", type=float, default=None)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    assert (args.data_dir is not None) ^ (args.xyz_file is not None), "Provide exactly one of --data_dir or --xyz_file"

    if args.data_dir:
        structs = load_xyz_directory(args.data_dir, pattern=args.pattern)
    else:
        structs = load_xyz_file(args.xyz_file)

    if not structs:
        raise SystemExit("No structures found.")

    ids = [s["id"] for s in structs]
    energies = np.array([float(s["energy"]) for s in structs], float)
    Emin = float(np.min(energies))
    deltaE = energies - Emin

    fps = compute_fingerprints(structs)
    group_id = group_by_dbscan(fps, min_samples=args.dbscan_min_samples, eps=args.dbscan_eps)
    is_outlier = mark_outliers(deltaE, z=3.0)
    buckets = quantile_buckets(deltaE, q=(0.25,0.75))

    df = pd.DataFrame({
        "id": ids,
        "E": energies,
        "DeltaE": deltaE,
        "group_id": group_id,
        "is_outlier": is_outlier,
        "energy_bucket": buckets,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "energies.csv"), index=False)

    # fixed test split
    train_ids, test_ids = stratified_group_test_split(df, test_frac=args.test_frac, seed=RANDOM_SEED)
    with open(os.path.join(args.out_dir, "test_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"test_ids": test_ids}, f, indent=2)

    # cv folds on remaining train pool
    df_trainpool = df[df.id.isin(train_ids)].copy()
    folds = stratified_group_kfold(df_trainpool, n_splits=5, seed=RANDOM_SEED)
    with open(os.path.join(args.out_dir, "cv_folds.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": folds}, f, indent=2)

    summary = {
        "num_structures": int(len(df)),
        "num_groups": int(df.group_id.nunique()),
        "emin": Emin,
        "E_mean": float(df["E"].mean()),
        "E_std": float(df["E"].std()),
        "DeltaE_mean": float(df["DeltaE"].mean()),
        "DeltaE_std": float(df["DeltaE"].std()),
        "bucket_counts": df["energy_bucket"].value_counts().to_dict(),
        "outliers_high_z3": int(df["is_outlier"].sum()),
        "test_size": int(len(test_ids)),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Preprocessing complete. Outputs saved to:", args.out_dir)

if __name__ == "__main__":
    main()