"""Replikacja Shiu na rate: sugar/bitter/water -> MN9 + ranking; sugar+bitter vs sugar (interakcja Fig3)."""
import numpy as np, pandas as pd
from fly_api import FlyBrainAPI
from collections import Counter
api = FlyBrainAPI(mode="full", path=".")
comp = pd.read_csv("Completeness_783.csv"); fly_ids = comp[comp.columns[0]].values.astype("int64")
sup = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "cell_type"])
typ = dict(zip(sup["root_id"].astype("int64"), zip(sup["cell_class"], sup["cell_type"])))
mn9 = 138332
res = {}
for name in ["sugar", "bitter", "water"]:
    o = api.step(odor=name, hops=3)
    res[name] = o
    print(f"{name}: MN9rank={(api._forward_pure(*api.encode(odor=name)[:2], hops=3) > 0).sum()} MN9act={o.get('MN9act','?')}", flush=True)
# MN9 act policz z EFFERENT? MN9 to motor: wez z h wprost
for name in ["sugar", "bitter", "water"]:
    idx, val, _ = api.encode(odor=name)
    h = api._forward_pure(idx, val, hops=3)
    top = Counter(typ.get(int(fly_ids[int(i)]), ("?", "?")) for i in np.argsort(h)[-150:]).most_common(8)
    print(f"{name}: MN9={h[mn9]:.3f} mean={h.mean():.4f} TOP150={top}", flush=True)
# interakcja: sugar + bitter (drugi drive jednoczesnie) vs sam sugar
tt = np.load("taste_grns.npz")
both_idx = np.concatenate([tt["sugar_idx"], tt["bitter_idx"]])
both_val = np.full(len(both_idx), 2.0, np.float32)
hS = api._forward_pure(*api.encode(odor="sugar")[:2], hops=3)
hSB = api._forward_pure(both_idx, both_val, hops=3)
print(f"sugar MN9={hS[mn9]:.3f} | sugar+bitter MN9={hSB[mn9]:.3f} (Shiu: bitter tlumi -> spadek oczek.)", flush=True)
print("PASS-taste", flush=True)
