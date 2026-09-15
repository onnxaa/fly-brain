"""Canonical DoOR 2.0 odors (Munch & Galizia 2016): glom -> ORN globals. Baseline 0.1, primary 1.0."""
import pandas as pd, numpy as np
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "cell_type"])
o = a[a["cell_class"] == "olfactory"]
by_type = {}
for t, r in zip(o["cell_type"], o["root_id"]):
    if int(r) in fly2idx:
        by_type.setdefault(t, []).append(fly2idx[int(r)])
CANON = {
    "geosmin": {"ORN_DA2": 1.0},            # Or56a, awersyjna linia (Stensmyr/DoOR)
    "co2": {"ORN_V": 1.0},                  # Gr21a/Gr63a ab1C
    "hexanone3": {"ORN_DM1": 1.0},          # Or42b
    "methyl_salicylate": {"ORN_DL1": 1.0},  # Or10a
    "butanedione": {"ORN_VA2": 1.0},        # Or92a
    "ethyl_hexanoate": {"ORN_DM2": 1.0},    # Or22a
}
out = {}
for name, prof in CANON.items():
    idx, val = [], []
    for t, ids in by_type.items():
        v = prof.get(t, 0.0)
        idx.extend(ids); val.extend([v]*len(ids))
    out[name+"_idx"] = np.array(idx, dtype=np.int32)
    out[name+"_val"] = np.array(val, dtype=np.float32)
np.savez("door_odors.npz", **out)
print(f"wrote {len(CANON)} odors; ORN types: {len(by_type)}", flush=True)
