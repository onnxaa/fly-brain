"""Olfactory laterality (side from supp1.tsv). Appends to roles.
Pattern like retmap.py: loads roles_full.npz as dict, appends, saves.
EXISTING keys (DESC_L/R, EFFERENT_L/R - from soma_x median) NOT touched (compat).
New: ORN_L/ORN_R (+ORN_U: 30 unassigned), ALPN_L/ALPN_R, MECH_L/MECH_R.
"""
import pandas as pd, numpy as np
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "side"])
a = a[a["root_id"].astype(np.int64).isin(fly2idx)]
roles = dict(np.load("roles_full.npz"))

def split740(mask):
    s = a[mask]
    L = np.array(sorted({fly2idx[int(r)] for r in s[s["side"] == "left"]["root_id"]}), dtype=np.int32)
    R = np.array(sorted({fly2idx[int(r)] for r in s[s["side"] == "right"]["root_id"]}), dtype=np.int32)
    U = np.array(sorted({fly2idx[int(r)] for r in s[~s["side"].isin(["left", "right"])]["root_id"]}), dtype=np.int32)
    return L, R, U

for cls, key in [("olfactory", "ORN"), ("ALPN", "ALPN"), ("mechanosensory", "MECH")]:
    L, R, U = split740(a["cell_class"] == cls)
    base = roles[key]
    assert len(L) + len(R) + len(U) == len(base), (key, len(L), len(R), len(U), len(base))
    assert len(set(L) & set(R)) == 0 and len(set(L) & set(U)) == 0 and len(set(R) & set(U)) == 0
    assert set(L) | set(R) | set(U) == set(int(x) for x in base), key
    roles[key + "_L"], roles[key + "_R"] = L, R
    if len(U):
        roles[key + "_U"] = U
    print(f"{key}: L={len(L)} R={len(R)} U={len(U)} (sum={len(base)})", flush=True)
np.savez("roles_full.npz", **roles)
print("PASS-laterality", flush=True)
