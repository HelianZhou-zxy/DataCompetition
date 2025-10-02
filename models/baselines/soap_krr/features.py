# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import List
from ase import Atoms
from dscribe.descriptors import SOAP as _SOAP  # 避免命名冲突

def build_soap_descriptor(
    species=("Au",),
    rcut=5.0,
    nmax=8,
    lmax=6,
    sigma=0.4,
    sparse=False,
    rbf="gto",
    periodic=False,
):
    """
    兼容多版本 dscribe 的 SOAP 构造器：
    1) 首选位置参数（species, rcut, nmax, lmax, sigma）
    2) 回退到新关键词 (rcut/nmax/lmax)
    3) 再回退到旧关键词 (r_cut/n_max/l_max)
    """
    try:
        return _SOAP(list(species), float(rcut), int(nmax), int(lmax), float(sigma),
                     rbf=rbf, periodic=periodic, sparse=sparse)
    except TypeError:
        try:
            return _SOAP(species=list(species), rcut=float(rcut), nmax=int(nmax), lmax=int(lmax),
                         sigma=float(sigma), rbf=rbf, periodic=periodic, sparse=sparse)
        except TypeError:
            return _SOAP(species=list(species), r_cut=float(rcut), n_max=int(nmax), l_max=int(lmax),
                         sigma=float(sigma), rbf=rbf, periodic=periodic, sparse=sparse)

def _to_numpy(x):
    # 稀疏 -> 稠密；list/tuple -> ndarray
    if hasattr(x, "toarray"):  # scipy.sparse
        x = x.toarray()
    return np.asarray(x)

def structure_soap_vector(soap: _SOAP, atoms: Atoms, pooling: str="sum") -> np.ndarray:
    """
    兼容多版本 create()：优先带 positions，失败则不带；同时兜底 1D/2D 返回。
    pooling: "sum" | "mean"
    """
    # 1) 先尝试带 positions
    try:
        X_local = soap.create(atoms, positions=list(range(len(atoms))))
    except TypeError:
        # 2) 某些版本不接受 positions 关键字：直接 create(atoms)
        X_local = soap.create(atoms)

    X_local = _to_numpy(X_local)
    if X_local.ndim == 1:
        # 某些配置（平均/全局描述符）直接返回 1D；此时 pool 无意义，直接返回
        return X_local

    # 2D: [n_atoms, feat_dim]
    if pooling == "sum":
        return np.sum(X_local, axis=0)
    elif pooling == "mean":
        return np.mean(X_local, axis=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

def batch_soap_vectors(soap: _SOAP, atoms_list: List[Atoms], pooling: str="sum") -> np.ndarray:
    return np.stack([structure_soap_vector(soap, a, pooling) for a in atoms_list], axis=0)
