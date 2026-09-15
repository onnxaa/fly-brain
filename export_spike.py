"""Second export: spiking-mode anatomy (APL/graded sets, mb APL extension).

Same construction as FlyBrainAPI._spike_ensure_caches, streamed in
batches (low RAM). Writes into ../fly_cpp/data/.
"""
import os
import re
import numpy as np
import pandas as pd

SRC = "."
DST = "../fly_cpp/data"


def w(name, arr):
    arr = np.ascontiguousarray(arr)
    ext = {"int32": "i32", "float32": "f32", "int64": "i64"}[str(arr.dtype)]
    arr.tofile(f"{DST}/{name}.{ext}")
    print(f"{name}: n={arr.size}", flush=True)


comp = pd.read_csv(f"{SRC}/Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
f2i = {int(f): i for i, f in enumerate(fly_ids)}
sup = pd.read_csv(f"{SRC}/supp1.tsv", sep="\t",
                  usecols=["root_id", "cell_type"])
apl_roots = sup[sup["cell_type"] == "APL"]["root_id"].astype(int).values
apl_g = np.array(sorted({f2i[int(r)] for r in apl_roots if int(r) in f2i}),
                 dtype=np.int32)
w("spike_apl", apl_g)
typ = dict(zip(sup["root_id"].astype(int), sup["cell_type"]))
gpat = re.compile(r"^(R1-6|R7|R8|Lai|L[1-5]|Dm\d+|DmDRA\d+|Mi[149]|"
                  r"Tm[1-49]|Tm20|C[23])$")
gset = np.array(sorted({f2i[int(r)] for r, t in typ.items()
                        if isinstance(t, str) and gpat.match(t)
                        and int(r) in f2i}), dtype=np.int32)
w("spike_gset", gset)
del sup, typ

# mb APL extension (test_lif v2): KC->APL and APL->KC edges, mb-local ids
import pyarrow.parquet as pq
d = np.load(f"{SRC}/mb_circuit.npz")
loc2glob = np.array([f2i[int(f)] for f in d["flywire_ids"]], dtype=np.int32)
glob2loc = {int(gg): int(ll) for ll, gg in enumerate(loc2glob)}
apl_g_mb = [f2i[int(r)] for r in apl_roots if int(r) in f2i]
apl_id = {g: k for k, g in enumerate(apl_g_mb)}
pf = pq.ParquetFile(f"{SRC}/Connectivity_783.parquet")
k2a_pre, k2a_w, k2a_apl = [], [], []
a2k_post, a2k_w, a2k_apl = [], [], []
apl_set = set(apl_g_mb)
loc_set = set(glob2loc)
for b in pf.iter_batches(batch_size=2000000,
                         columns=["Presynaptic_Index", "Postsynaptic_Index",
                                  "Excitatory x Connectivity"]):
    pr = np.asarray(b["Presynaptic_Index"]).astype(np.int64)
    po = np.asarray(b["Postsynaptic_Index"]).astype(np.int64)
    ww = np.abs(np.asarray(b["Excitatory x Connectivity"]).astype(float))
    m = np.array([(q in apl_set) and (int(p) in loc_set)
                  for p, q in zip(pr, po)])
    if m.any():
        k2a_pre.append([glob2loc[int(p)] for p in pr[m]])
        k2a_w.append(ww[m])
        k2a_apl.append([apl_id[int(q)] for q in po[m]])
    m2 = np.array([(int(p) in apl_set) and (int(q) in loc_set)
                   for p, q in zip(pr, po)])
    if m2.any():
        a2k_post.append([glob2loc[int(q)] for q in po[m2]])
        a2k_w.append(ww[m2])
        a2k_apl.append([apl_id[int(p)] for p in pr[m2]])
w("spike_k2a_pre", np.concatenate(k2a_pre).astype(np.int32))
w("spike_k2a_w", np.concatenate(k2a_w).astype(np.float64).astype(np.float32))
w("spike_k2a_apl", np.concatenate(k2a_apl).astype(np.int32))
w("spike_a2k_post", np.concatenate(a2k_post).astype(np.int32))
w("spike_a2k_w", np.concatenate(a2k_w).astype(np.float64).astype(np.float32))
w("spike_a2k_apl", np.concatenate(a2k_apl).astype(np.int32))
print("OK", flush=True)
