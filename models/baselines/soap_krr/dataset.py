# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob
import numpy as np
from typing import List, Tuple
from ase import Atoms

FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|\d+")

def _parse_one_xyz_block(lines: List[str]) -> Tuple[Atoms, float]:
    nat = int(lines[0].strip())
    comment = lines[1].strip()
    m = re.search(r"Energy\s*=\s*([-\d\.Ee+]+)|\bE\s*=\s*([-\d\.Ee+]+)", comment)
    if m:
        energy = float(m.group(1) or m.group(2))
    else:
        m2 = FLOAT_RE.search(comment)
        if not m2: raise ValueError(f"No energy in xyz comment: {comment!r}")
        energy = float(m2.group(0))
    symbols, pos = [], []
    for k in range(nat):
        parts = lines[2+k].split()
        symbols.append(parts[0])
        pos.append(list(map(float, parts[1:4])))
    return Atoms(symbols=symbols, positions=np.asarray(pos), pbc=False), float(energy)

def load_xyz_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip().splitlines()
    i, L = 0, len(raw)
    atoms_list, energies = [], []
    while i < L:
        nat = int(raw[i].strip())
        block = raw[i:i+2+nat]
        if len(block) < 2+nat: break
        a, E = _parse_one_xyz_block(block)
        atoms_list.append(a); energies.append(E)
        i += 2+nat
    if not atoms_list:
        a, E = _parse_one_xyz_block(raw)
        atoms_list, energies = [a], [E]
    return atoms_list, np.asarray(energies, dtype=float)

def load_xyz_from_dir(dir_path: str, pattern: str="*.xyz"):
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    atoms_list, energies = [], []
    for fp in files:
        a_list, e_arr = load_xyz_from_file(fp)
        if len(a_list) != 1:
            raise ValueError(f"Expect 1-frame xyz per file, got {len(a_list)} in {fp}")
        atoms_list.append(a_list[0]); energies.append(float(e_arr[0]))
    return atoms_list, np.asarray(energies, dtype=float)
