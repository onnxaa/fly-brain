"""Wersja light bez torch: zapisuje mb_circuit.npz (numpy)."""
import json
import numpy as np
import pandas as pd

print("loading completeness...", flush=True)
comp = pd.read_csv("Completeness_783.csv")
id_col = comp.columns[0]
fly_ids = comp[id_col].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
print(f"neurons: {len(fly_ids)}", flush=True)

print("loading annotations...", flush=True)
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class"])
a["flywire_id"] = a["root_id"].astype(np.int64)
a = a[a["flywire_id"].isin(fly2idx)]
a["idx"] = a["flywire_id"].map(fly2idx)
kc = set(a[a["cell_class"] == "Kenyon_Cell"]["idx"])
mbon = set(a[a["cell_class"] == "MBON"]["idx"])
dan = set(a[a["cell_class"] == "DAN"]["idx"])
alpn = set(a[a["cell_class"] == "ALPN"]["idx"])
print(f"KC={len(kc)} MBON={len(mbon)} DAN={len(dan)} ALPN={len(alpn)}", flush=True)
keep = kc | mbon | dan | alpn
print(f"keep neurons: {len(keep)}", flush=True)

print("loading connectivity (cols)...", flush=True)
con = pd.read_parquet("Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
print(f"edges: {len(con)}", flush=True)
sub = con[con["Presynaptic_Index"].isin(keep) & con["Postsynaptic_Index"].isin(keep)]
print(f"keep edges: {len(sub)} ({len(sub)/len(con)*100:.1f}%)", flush=True)
w = sub["Excitatory x Connectivity"].values.astype(np.float32)
print(f"w: min={w.min()} max={w.max()} mean={w.mean():.3f} sum={w.sum():.1f} pos={(w>0).sum()} neg={(w<0).sum()} zero={(w==0).sum()}", flush=True)
# histogram progowy
for thr in [1,2,5,10,20,50,100]:
    print(f"  |w|>={thr}: {(np.abs(w)>=thr).sum()}", flush=True)

nodes = sorted(keep)
loc = {g: l for l, g in enumerate(nodes)}
idx2fly = np.array([int(fly_ids[g]) for g in nodes], dtype=np.int64)
pre = np.array([loc[i] for i in sub["Presynaptic_Index"].values], dtype=np.int32)
post = np.array([loc[i] for i in sub["Postsynaptic_Index"].values], dtype=np.int32)
roles = {
    "inputs_ALPN": np.array(sorted(loc[i] for i in alpn), dtype=np.int32),
    "KC": np.array(sorted(loc[i] for i in kc), dtype=np.int32),
    "MBON": np.array(sorted(loc[i] for i in mbon), dtype=np.int32),
    "DAN": np.array(sorted(loc[i] for i in dan), dtype=np.int32),
}
np.savez_compressed("mb_circuit.npz", pre=pre, post=post, weight=w,
                    flywire_ids=idx2fly,
                    inputs_ALPN=roles["inputs_ALPN"], KC=roles["KC"],
                    MBON=roles["MBON"], DAN=roles["DAN"])
print("saved mb_circuit.npz", flush=True)
print({k: len(v) for k, v in roles.items()}, flush=True)
