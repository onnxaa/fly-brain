"""Wzorce ALPN z wag ORN->ALPN dla zapachow DoOR (do wstrzykniecia w Brian-MB)."""
import numpy as np, pandas as pd
door = np.load("door_odors.npz")
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
mb = np.load("mb_circuit.npz"); mb_fly = mb["flywire_ids"]
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
ALPN_glob = set(int(x) for x in np.sort(loc2glob[mb["inputs_ALPN"]]))
con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values; post = con["Postsynaptic_Index"].values
w = con["Excitatory x Connectivity"].values.astype(float); del con
m = np.array([(p not in ALPN_glob) and (q in ALPN_glob) for p, q in zip(pre, post)])
print(f"krawedzi ->ALPN spoza ALPN: {m.sum()}", flush=True)
out = {}
base_corr = {}
for name in ["geosmin", "hexanone3", "co2", "butanedione", "methyl_salicylate", "ethyl_hexanoate"]:
    drv = dict(zip(door[name+"_idx"].tolist(), door[name+"_val"].tolist()))
    acc = {}
    for a, b, c in zip(pre[m], post[m], w[m]):
        d = drv.get(int(a), 0.0)
        if d:
            acc[int(b)] = acc.get(int(b), 0) + d*max(c, 0)
    v = np.array([acc.get(g, 0) for g in sorted(ALPN_glob)])
    out[name] = v
    print(f"{name}: ALPN aktywnych={(v>0).sum()}/{len(v)} max={v.max():.1f}", flush=True)
import itertools
for x, y in itertools.combinations(["geosmin", "hexanone3", "co2", "butanedione", "methyl_salicylate", "ethyl_hexanoate"], 2):
    print(f"corr({x},{y})={np.corrcoef(out[x], out[y])[0,1]:.2f}", flush=True)
np.savez("alpn_patterns.npz", **out, ALPN_glob=np.array(sorted(ALPN_glob)))
print("PASS-alpn-pat", flush=True)
