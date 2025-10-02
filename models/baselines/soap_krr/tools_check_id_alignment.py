# -*- coding: utf-8 -*-
import os, sys, glob, pandas as pd

energies_csv = r"D:\DataComp\DataCompetition\data\preproc_out\energies.csv"
xyz_dir      = r"D:\DataComp\DataCompetition\data\raw\Au20_OPT_1000"

df = pd.read_csv(energies_csv)
df['id'] = df['id'].astype(str)

# 目录里的文件名（不含扩展名）
file_ids = set(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(xyz_dir, "*.xyz")))

ids = list(df['id'])
ids_in = [i for i in ids if i in file_ids]
ids_out= [i for i in ids if i not in file_ids]

print(f"[OK in geom] {len(ids_in)} / {len(ids)}")
print(f"[MISSING in geom] {len(ids_out)}")
print(ids_out[:20])

# 如果 energies.csv 里的 id 是 "xxx#0" 这种，建议生成一个“去#后缀”的版本
if len(ids_out) > 0 and any('#' in i for i in ids):
    df2 = df.copy()
    df2['id'] = df2['id'].str.split('#').str[0]
    out = r"D:\DataComp\DataCompetition\data\preproc_out\energies_nohash.csv"
    df2.to_csv(out, index=False)
    print(f"[Wrote] normalized energies to: {out}")
