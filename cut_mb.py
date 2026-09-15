"""Cuts the real mushroom-body circuit out of FlyWire v783.
Inputs (no token, all public):
  Connectivity_783.parquet  (philshiu/Drosophila_brain_model)
  Completeness_783.csv      (FlyWireID -> index 0..N-1)
  supp1.tsv                 (flyconnectome/flywire_annotations, v783)
Output: mb_circuit.pt with edge_index/weight in local indices 0..M-1
  + roles: inputs (ALPN), KC, MBON (readout), DAN (reward).
"""
import json
import pandas as pd
import torch

print("loading connectivity...")
con = pd.read_parquet("Connectivity_783.parquet")
print("edges:", len(con), con.columns.tolist())

print("loading completeness...")
comp = pd.read_csv("Completeness_783.csv")
# file header is ",Completed" -> first column is the FlyWire ID
id_col = comp.columns[0]
comp = comp.rename(columns={id_col: "flywire_id"})
comp["flywire_id"] = comp["flywire_id"].astype("int64")
fly2idx = {int(f): i for i, f in enumerate(comp["flywire_id"].values)}
print("neurons:", len(comp))

print("loading annotations...")
a = pd.read_csv("supp1.tsv", sep="\t",
                usecols=["root_id", "cell_class", "cell_type"],
                dtype={"root_id": str})
a["flywire_id"] = a["root_id"].astype(int)
a = a[a["flywire_id"].isin(fly2idx)]
a["idx"] = a["flywire_id"].map(fly2idx)

kc = set(a[a["cell_class"] == "Kenyon_Cell"]["idx"])
mbon = set(a[a["cell_class"] == "MBON"]["idx"])
dan = set(a[a["cell_class"] == "DAN"]["idx"])
alpn = set(a[a["cell_class"] == "ALPN"]["idx"])
print(f"KC={len(kc)} MBON={len(mbon)} DAN={len(dan)} ALPN={len(alpn)}")

keep = kc | mbon | dan | alpn
print("keep neurons:", len(keep))

sub = con[con["Presynaptic_Index"].isin(keep) & con["Postsynaptic_Index"].isin(keep)]
print("keep edges:", len(sub), f"({len(sub)/len(con)*100:.1f}% wszystkich)")

w = sub["Excitatory x Connectivity"].values
print("w: mean=%.3f sum=%.1f pos=%d neg=%d" % (
    w.mean(), w.sum(), (w > 0).sum(), (w < 0).sum()))

# remap do 0..M-1
nodes = sorted(keep)
loc = {g: l for l, g in enumerate(nodes)}
idx2fly = [int(comp["flywire_id"].values[g]) for g in nodes]
pre = torch.tensor([loc[i] for i in sub["Presynaptic_Index"].values], dtype=torch.long)
post = torch.tensor([loc[i] for i in sub["Postsynaptic_Index"].values], dtype=torch.long)
edge_index = torch.stack([pre, post])
weight = torch.tensor(w, dtype=torch.float32)

roles = {
    "inputs_ALPN": sorted(loc[i] for i in alpn),
    "KC": sorted(loc[i] for i in kc),
    "MBON": sorted(loc[i] for i in mbon),
    "DAN": sorted(loc[i] for i in dan),
}
torch.save({"edge_index": edge_index, "weight": weight,
            "flywire_ids": idx2fly, "roles": roles}, "mb_circuit.pt")
with open("mb_roles.json", "w") as f:
    json.dump({k: len(v) for k, v in roles.items()}, f, indent=1)
print("saved mb_circuit.pt, roles:", {k: len(v) for k, v in roles.items()})
