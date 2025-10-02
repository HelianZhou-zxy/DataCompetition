# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_hist(y, out_png, bins: int=50):
    import matplotlib; matplotlib.use("Agg")
    plt.figure(figsize=(5,4)); plt.hist(y, bins=bins, alpha=0.8)
    plt.xlabel("Energy (E or ΔE)"); plt.ylabel("Count"); plt.title("Energy distribution")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def parity_plot(y_true, y_pred, out_png):
    import matplotlib; matplotlib.use("Agg")
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(5,5)); plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], ls="--")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Parity plot")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def binned_mae(y_true, y_pred, out_png, n_bins: int=10):
    import matplotlib; matplotlib.use("Agg")
    q = np.linspace(0,1,n_bins+1); edges = np.unique(np.quantile(y_true, q))
    idx = np.digitize(y_true, edges[1:-1], right=False)
    maes, centers = [], []
    for b in range(len(edges)-1):
        m = (idx==b)
        if np.any(m):
            maes.append(np.mean(np.abs(y_true[m]-y_pred[m])))
            centers.append(np.mean([edges[b], edges[b+1]]))
    plt.figure(figsize=(6,4)); plt.plot(centers, maes, marker="o")
    plt.xlabel("Energy bin center"); plt.ylabel("MAE"); plt.title("Binned MAE")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
