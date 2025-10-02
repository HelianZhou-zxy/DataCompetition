# -*- coding: utf-8 -*-
"""
viz_stiffness.py — 用 per-atom stiffness K_i 做可视化：
1) 生成带 B-factor 的 PDB（或手写 PDB+CONEXT）；
2) 生成 VMD 脚本（按 Beta 上色，设置颜色范围）；
3) 生成 Matplotlib 3D 图（可选标注 top-k）。
"""

import os, sys, argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 让脚本能从项目根导入你之前的工具
HERE = Path(__file__).resolve()
# models/gnn/schnet -> parents: [schnet, gnn, models, <projroot>]
PROJ_ROOT = HERE.parents[3]
sys.path.append(str(PROJ_ROOT))

from models.baselines.soap_krr.dataset import load_xyz_from_file  # noqa: E402

def parse_args():
    ap = argparse.ArgumentParser("Visualize per-atom stiffness K_i on Au20")
    ap.add_argument("--xyz_file", required=True)
    ap.add_argument("--site_csv", required=True, help="CSV with columns: atom_index, K_i")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cutoff", type=float, default=3.0, help="draw bonds if d<=cutoff (Å)")
    ap.add_argument("--colormap", type=str, default="viridis")
    ap.add_argument("--label_topk", type=int, default=5, help="label top-K stiff & soft atoms in 3D plot")
    return ap.parse_args()

def _as_posix(p):
    return str(Path(p).resolve().as_posix())

def compute_bonds(positions, cutoff):
    N = positions.shape[0]
    bonds = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= cutoff:
                bonds.append((i, j))
    return bonds

def write_pdb_simple(out_path, positions, K, element="AU", bonds=None):
    """
    生成一个“自写”的最朴素 PDB，occupancy=1.00，B-factor=K_i；
    bonds 会被写成 CONECT 行，VMD 可以直接识别。
    """
    with open(out_path, "w", encoding="ascii") as f:
        for i, (x, y, z) in enumerate(positions, start=1):
            b = float(K[i-1])
            # PDB ATOM 行（列宽格式）
            line = (
                f"ATOM  {i:5d} {element:>2s}   CLT A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b:6.2f}          {element:>2s}\n"
            )
            f.write(line)
        if bonds:
            for (i, j) in bonds:
                f.write(f"CONECT{i+1:5d}{j+1:5d}\n")
        f.write("END\n")

def make_vmd_tcl(pdb_path, out_tcl, kmin, kmax):
    """
    生成最小 VMD 脚本：读取 PDB，按 Beta（即 B-factor）着色，设置色标范围。
    避免任何带反斜杠的表达式，全部转成 POSIX 路径。
    """
    pdb_posix = _as_posix(pdb_path)
    tcl = "\n".join([
        f'mol new "{pdb_posix}" type pdb',
        'mol delrep 0 top',
        'mol representation VDW 1.2 12.0',
        'mol color Beta',
        'mol selection all',
        'mol material Opaque',
        'mol addrep top',
        # 设置颜色范围（第 0 个 representation）
        f'mol scaleminmax top 0 {kmin:.6f} {kmax:.6f}',
        'display projection Orthographic',
        'axes location Off',
    ])
    Path(out_tcl).write_text(tcl, encoding="utf-8")

def plot_3d(points, K, bonds, cm_name, label_topk, out_png):
    pts = points
    Ki = np.asarray(K, dtype=float)
    cm = plt.get_cmap(cm_name)
    norm = plt.Normalize(vmin=np.min(Ki), vmax=np.max(Ki))
    colors = cm(norm(Ki))

    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=120, c=colors, depthshade=True, edgecolors="k", linewidths=0.5)

    # 画键
    for i, j in bonds:
        xs = [pts[i, 0], pts[j, 0]]
        ys = [pts[i, 1], pts[j, 1]]
        zs = [pts[i, 2], pts[j, 2]]
        ax.plot(xs, ys, zs, alpha=0.6)

    # 标注 top-k stiff & soft
    if label_topk > 0:
        idx_desc = np.argsort(-Ki)[:label_topk]
        idx_asc  = np.argsort(Ki)[:label_topk]
        for idx in idx_desc:
            ax.text(pts[idx, 0], pts[idx, 1], pts[idx, 2], f"{idx}", color="red", fontsize=9, weight="bold")
        for idx in idx_asc:
            ax.text(pts[idx, 0], pts[idx, 1], pts[idx, 2], f"{idx}", color="blue", fontsize=9, weight="bold")

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("Per-atom stiffness $K_i$")
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("$K_i$")

    # 视角稍微调一下
    ax.view_init(elev=15, azim=35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读几何
    atoms_list, _ = load_xyz_from_file(args.xyz_file)
    at = atoms_list[0]
    pos = np.asarray(at.get_positions(), dtype=float)
    N = pos.shape[0]

    # 读 per-atom K_i
    df = pd.read_csv(args.site_csv)
    if not {"atom_index", "K_i"}.issubset(df.columns):
        raise ValueError("site_csv 必须包含列: atom_index, K_i")
    df = df.sort_values("atom_index")
    K = df["K_i"].to_numpy(dtype=float)
    if len(K) != N:
        raise ValueError(f"K_i 数量({len(K)})与原子数({N})不一致")

    # 计算简单键（用于 VMD/Matplotlib 画连线）
    bonds = compute_bonds(pos, cutoff=args.cutoff)

    # 写 PDB（手写版本最稳）
    pdb_path = out_dir / "au20_Ki.pdb"
    write_pdb_simple(pdb_path, pos, K, element="AU", bonds=bonds)

    # 写 VMD 脚本
    kmin, kmax = float(np.min(K)), float(np.max(K))
    tcl_path = out_dir / "au20_Ki_vmd.tcl"
    make_vmd_tcl(pdb_path, tcl_path, kmin, kmax)

    # Matplotlib 3D
    png_path = out_dir / "au20_Ki_3D.png"
    plot_3d(pos, K, bonds, args.colormap, args.label_topk, png_path)

    # 输出 top/bottom 列表
    topk = int(args.label_topk) if args.label_topk > 0 else min(5, N)
    idx_desc = np.argsort(-K)[:topk]
    idx_asc  = np.argsort(K)[:topk]
    df_tb = pd.DataFrame({
        "type": ["top_stiff"]*topk + ["bottom_soft"]*topk,
        "atom_index": np.concatenate([idx_desc, idx_asc]),
        "K_i": np.concatenate([K[idx_desc], K[idx_asc]])
    })
    df_tb.to_csv(out_dir / "au20_Ki_top_bottom_sites.csv", index=False)

    print("[Saved]", str(pdb_path))
    print("[Saved]", str(tcl_path))
    print("[Saved]", str(png_path))
    print("[Saved]", str(out_dir / "au20_Ki_top_bottom_sites.csv"))
    print(f"[Range] K_i: min={kmin:.4f}, max={kmax:.4f}")

if __name__ == "__main__":
    main()
