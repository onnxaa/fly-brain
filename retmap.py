"""Mapa retinotopowa VIS (pozycje z supp1, dwa oczy) + grupy ME/LO. Dopisuje do ról."""
import pandas as pd, numpy as np
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "pos_x", "pos_y"])
roles = dict(np.load("roles_full.npz"))
VIS = roles["VIS"]
pos = {int(r): (x, y) for r, x, y in zip(a["root_id"], a["pos_x"], a["pos_y"])}
vx = np.array([pos.get(int(f), (0, 0))[0] for f in [0]*0])  # placeholder
# VIS globals -> flywire -> pos
idx2fly = fly_ids[np.array(VIS)]
px = np.array([pos.get(int(f), (np.nan, np.nan))[0] for f in idx2fly], dtype=float)
py = np.array([pos.get(int(f), (np.nan, np.nan))[1] for f in idx2fly], dtype=float)
med = np.nanmedian(px); eye = (px > med).astype(int)  # 0/1 - strony
pix = np.zeros(len(VIS), dtype=np.int32)
# KAZDE oko widzi swoje hemipole: oko0 <- kolumny 0..31, oko1 <- kolumny 32..63.
# Wiersz z pos_y (64), pozycja w hemipolu z pos_x (32).
for e in (0, 1):
    m = (eye == e) & ~np.isnan(px) & ~np.isnan(py)
    ry = pd.qcut(py[m], 64, labels=False, duplicates="drop")
    rx = pd.qcut(px[m], 32, labels=False, duplicates="drop")
    pix[m] = (np.asarray(ry)*64 + (e*32 + np.asarray(rx))).astype(np.int32)
roles["VIS_pix"] = pix
roles["VIS_eye"] = eye.astype(np.int32)
for cc in ["ME", "LO"]:
    m = a[a["cell_class"] == cc]
    idx = np.array(sorted({fly2idx[int(r)] for r in m["root_id"] if int(r) in fly2idx}), dtype=np.int32)
    roles[cc] = idx
np.savez("roles_full.npz", **roles)
print(f"VIS {len(VIS)} oczy: L={(eye==0).sum()} R={(eye==1).sum()} ME {len(roles['ME'])} LO {len(roles['LO'])}", flush=True)
print("PASS-retmap", flush=True)
