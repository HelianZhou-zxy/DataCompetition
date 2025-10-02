"""
geometry_fingerprints.py
------------------------
Geometry helpers for Au20 preprocessing.

Includes:
- pairwise_distances_fingerprint: sorted condensed distance vector (permutation/rotation/translation invariant)
- distance_histogram: binned histogram of pair distances
- radial_distribution + first minimum finder (to choose rc)
- soft_cn: smooth coordination number with logistic switching
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

def pairwise_distances_fingerprint(coords: np.ndarray) -> np.ndarray:
    """Return sorted (i<j) pairwise distances as a 1D vector of length N*(N-1)/2."""
    N = coords.shape[0]
    d = []
    for i in range(N-1):
        dv = coords[i+1:] - coords[i]
        di = np.linalg.norm(dv, axis=1)
        d.append(di)
    v = np.concatenate(d, axis=0)
    return np.sort(v)

def distance_histogram(coords: np.ndarray, bins: int=64, r_max: float=None):
    vec = pairwise_distances_fingerprint(coords)
    if r_max is None:
        r_max = float(np.percentile(vec, 99.5))
    hist, edges = np.histogram(vec, bins=bins, range=(0.0, r_max), density=True)
    return hist.astype(np.float32), edges

def radial_distribution(struct_coords: List[np.ndarray], bins: int=200, r_max: float=None):
    """Histogram of pair distances pooled over many structures (finite cluster variant)."""
    allv = []
    for c in struct_coords:
        allv.append(pairwise_distances_fingerprint(c))
    allv = np.concatenate(allv, axis=0)
    if r_max is None:
        r_max = float(np.percentile(allv, 99.9))
    g, edges = np.histogram(allv, bins=bins, range=(0.0, r_max), density=True)
    return g.astype(np.float32), edges

def first_minimum_after_first_peak(y: np.ndarray, edges: np.ndarray) -> float:
    """A simple robust heuristic to find the first minimum after the first peak in histogram y."""
    centers = 0.5*(edges[:-1] + edges[1:])
    peak = int(np.argmax(y))
    for i in range(peak+1, len(y)-2):
        if y[i] < y[i-1] and y[i] <= y[i+1] and y[i+1] < y[i+2]:
            return float(centers[i])
    j = peak + 1 + int(np.argmin(y[peak+1:]))
    return float(centers[j])

def soft_cn(coords: np.ndarray, rc: float, alpha: float=10.0) -> np.ndarray:
    """Soft coordination numbers with logistic switching centered at rc."""
    N = coords.shape[0]
    cn = np.zeros(N, float)
    for i in range(N):
        dv = coords - coords[i]
        dist = np.linalg.norm(dv, axis=1)
        dist = dist[np.arange(N)!=i]
        s = 1.0/(1.0 + np.exp(alpha*(dist - rc)))
        cn[i] = s.sum()
    return cn