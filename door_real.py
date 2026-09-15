"""FINAL: prawdziwe profile DoOR (inkl. inhibicja) -> door_odors.npz."""
import pandas as pd, numpy as np
M = pd.read_csv("door_matrix.csv", sep=";", index_col=0)
mp = pd.read_csv("door_mappings.csv", sep=";")
unit2glom = {}
for _, r in mp.iterrows():
    g = str(r["glomerulus"])
    if g in ("?", "", "nan") or pd.isna(r["glomerulus"]):
        continue
    for col in ["receptor", "code", "code.OSN", "OSN"]:
        u = str(r[col])
        if u not in ("?", "", "nan") and not pd.isna(r[col]):
            unit2glom.setdefault(u, g)
od = pd.read_csv("door_odor.csv", sep=";").set_index("InChIKey")
KEYS = {}
for name, cas in [("geosmin", "16423-19-1"), ("co2", "124-38-9"), ("hexanone3", "589-38-8"),
                  ("methyl_salicylate", "119-36-8"), ("butanedione", "431-03-8"), ("ethyl_hexanoate", "123-66-0")]:
    hit = od[od["CAS"].astype(str) == cas]
    KEYS[name] = hit.index[0]
    print(name, "->", hit["Name"].values[0], flush=True)
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "cell_type"])
o = a[a["cell_class"] == "olfactory"]
glom2idx = {}
for t, r in zip(o["cell_type"], o["root_id"]):
    if int(r) in fly2idx:
        gl = str(t).replace("ORN_", "")
        glom2idx.setdefault(gl, []).append(fly2idx[int(r)])
out = {}
for name, key in KEYS.items():
    row = M.loc[key].astype(float)
    gprof = {}
    for unit, val in row.items():
        if pd.isna(val):
            continue
        g = unit2glom.get(str(unit))
        if g:
            gprof.setdefault(g, []).append(val)
    gmean = {g: float(np.mean(v)) for g, v in gprof.items()}
    mx = max([abs(v) for v in gmean.values()] + [1e-9])
    idx, val = [], []
    for gl, ids in glom2idx.items():
        v = np.clip(gmean.get(gl, 0.0)/mx, -1, 1)
        idx.extend(ids); val.extend([v]*len(ids))
    out[name+"_idx"] = np.array(idx, dtype=np.int32)
    out[name+"_val"] = np.array(val, dtype=np.float32)
    nz = sum(1 for gl in glom2idx if abs(gmean.get(gl, 0)) > 0.01)
    print(f"{name}: glomeruli aktywnych {nz}/{len(glom2idx)}, DA2={gmean.get('DA2',0):+.2f} DM1={gmean.get('DM1',0):+.2f} V={gmean.get('V',0):+.2f}", flush=True)
np.savez("door_odors.npz", **out)
print("saved REAL door_odors.npz", flush=True)
