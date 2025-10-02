# -*- coding: utf-8 -*-
"""
tools_normalize_splits.py
- 读取 energies_nohash.csv 获取合法 id（与几何目录求交集）
- 将 test_ids.json / cv_folds.json 中的 id 统一去掉 '#...' 后缀
- 过滤不存在的 id、去重、保持原相对顺序
- 输出 *_nohash.json
"""
import os, re, json, glob, argparse
import pandas as pd

def strip_hash(x: str) -> str:
    return x.split("#")[0]

def load_valid_ids(energies_csv: str, xyz_dir: str) -> set:
    df = pd.read_csv(energies_csv)
    df["id"] = df["id"].astype(str).map(strip_hash)
    ids_energy = set(df["id"].tolist())
    file_ids = set(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(xyz_dir, "*.xyz")))
    return ids_energy & file_ids

def dedup_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energies_nohash_csv", required=True)
    ap.add_argument("--xyz_dir", required=True)
    ap.add_argument("--in_test_json", required=True)
    ap.add_argument("--in_folds_json", required=True)
    ap.add_argument("--out_test_json", required=True)
    ap.add_argument("--out_folds_json", required=True)
    args = ap.parse_args()

    valid = load_valid_ids(args.energies_nohash_csv, args.xyz_dir)
    print(f"[valid ids] {len(valid)}")

    # test ids
    with open(args.in_test_json, "r", encoding="utf-8") as f:
        tj = json.load(f)
    raw_test = [strip_hash(str(i)) for i in tj.get("test_ids", [])]
    test_ids = [i for i in dedup_preserve(raw_test) if i in valid]
    print(f"[test_ids] {len(test_ids)} kept after normalize/filter")

    with open(args.out_test_json, "w", encoding="utf-8") as f:
        json.dump({"test_ids": test_ids}, f, indent=2, ensure_ascii=False)

    # folds
    with open(args.in_folds_json, "r", encoding="utf-8") as f:
        fj = json.load(f)
    out_folds = []
    for k, fold in enumerate(fj.get("folds", []), start=1):
        tr = [strip_hash(str(i)) for i in fold.get("train_ids", [])]
        va = [strip_hash(str(i)) for i in fold.get("val_ids",   [])]
        tr = [i for i in dedup_preserve(tr) if i in valid and i not in test_ids]
        va = [i for i in dedup_preserve(va) if i in valid and i not in test_ids]
        # 防交集
        tr = [i for i in tr if i not in set(va)]
        out_folds.append({"train_ids": tr, "val_ids": va})
        print(f"[fold {k}] train={len(tr)}, val={len(va)}")
    with open(args.out_folds_json, "w", encoding="utf-8") as f:
        json.dump({"folds": out_folds}, f, indent=2, ensure_ascii=False)

    # 简报
    print(f"[saved] {args.out_test_json}")
    print(f"[saved] {args.out_folds_json}")

if __name__ == "__main__":
    main()
