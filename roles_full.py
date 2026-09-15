"""Full-brain roles (lightweight, no connectivity load)."""
import numpy as np, pandas as pd
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
N = len(fly_ids)
print(f"N={N}", flush=True)
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "flow", "super_class", "cell_class", "cell_type"])
a["flywire_id"] = a["root_id"].astype(np.int64)
a = a[a["flywire_id"].isin(fly2idx)]
a["idx"] = a["flywire_id"].map(fly2idx)

def idxs(mask):
    return np.array(sorted(a[mask]["idx"].values), dtype=np.int32)

olf = idxs(a["cell_class"] == "olfactory")
mech = idxs(a["cell_class"] == "mechanosensory")
vis = idxs(a["cell_class"] == "visual")
alpn = idxs(a["cell_class"] == "ALPN")
aff = idxs(a["flow"] == "afferent")
eff = idxs(a["flow"] == "efferent")
desc = idxs(a["super_class"] == "descending")
mot = idxs(a["super_class"] == "motor")
dn_cls = idxs(a["cell_class"].isna() & a["cell_type"].str.contains("DN|MDN|Giant", na=False))
print(f"olf={len(olf)} mech={len(mech)} vis={len(vis)} ALPN={len(alpn)} afferent={len(aff)} efferent={len(eff)} descending={len(desc)} motor={len(mot)} DNtype={len(dn_cls)}", flush=True)
# MN9 check
mn9 = a[a["root_id"] == 720575940660219265]
print("MN9:", mn9[["idx", "flow", "super_class", "cell_class", "cell_type"]].to_string(), flush=True)
np.savez("roles_full.npz", ORN=olf, MECH=mech, VIS=vis, ALPN=alpn,
         AFFERENT=aff, EFFERENT=eff, DESC=desc, MOTOR=mot, N=np.array([N]))
print("saved roles_full.npz", flush=True)
