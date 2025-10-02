"""
dataset_io.py
--------------
Robust XYZ parser for the Au20 competition.

Supports:
- A directory of .xyz files (each file may contain one or many XYZ blocks)
- A single multi-structure .xyz file

Format assumptions (tolerant to extra tokens/spaces):
- Line 1 of each block: integer atom count (20 for Au20)
- Line 2 of each block: contains total energy; we take the *first* float on the line
- Next N lines: "Element x y z" (e.g., "Au 0.0 0.0 0.0")

Returns each structure as a dict:
{
  "id": "<file-stem>#<block_idx>",
  "energy": float,
  "coords": numpy.ndarray (N, 3),
  "elements": List[str] length N
}
"""

from __future__ import annotations
import pathlib
import numpy as np
import re
from typing import List, Tuple, Optional

_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def _first_float(text: str) -> Optional[float]:
    m = _FLOAT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _read_block(lines: List[str], idx: int) -> Tuple[Optional[dict], int]:
    """Read one XYZ block starting at index idx. Return (struct_or_None, next_idx)."""
    if idx >= len(lines):
        return None, len(lines)
    try:
        n = int(lines[idx].strip().split()[0])
    except Exception:
        return None, len(lines)
    i = idx + 1
    if i >= len(lines):
        return None, len(lines)
    e_line = lines[i].rstrip("\n")
    energy = _first_float(e_line)
    i += 1
    coords, elements = [], []
    for _ in range(n):
        if i >= len(lines):
            return None, len(lines)
        toks = lines[i].split()
        if len(toks) < 4:
            return None, len(lines)
        elements.append(toks[0])
        coords.append([float(toks[1]), float(toks[2]), float(toks[3])])
        i += 1
    return {"energy": float(energy), "coords": np.array(coords, float), "elements": elements}, i

def load_xyz_directory(dir_path: str, pattern: str="*.xyz") -> List[dict]:
    p = pathlib.Path(dir_path)
    files = sorted(p.rglob(pattern))
    out: List[dict] = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        idx, k = 0, 0
        while True:
            s, idx = _read_block(lines, idx)
            if s is None:
                break
            s["id"] = f"{f.stem}#{k}"
            out.append(s)
            k += 1
    return out

def load_xyz_file(file_path: str) -> List[dict]:
    p = pathlib.Path(file_path)
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    out: List[dict] = []
    idx, k = 0, 0
    while True:
        s, idx = _read_block(lines, idx)
        if s is None:
            break
        s["id"] = f"{p.stem}#{k}"
        out.append(s)
        k += 1
    return out