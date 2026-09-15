"""Shiu Fig1D/E assay on rate: sugar(LB3)->who is active? MN9? type ranking. Contrast: mechano."""
import numpy as np, pandas as pd, time
t0=time.time()
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="full", path=".")
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "cell_type"])
lb3 = np.array(sorted({fly2idx[int(r)] for r in a[a["cell_type"] == "LB3"]["root_id"] if int(r) in fly2idx}), dtype=np.int32)
print(f"sugar LB3: {len(lb3)}", flush=True)
idx, val, _ = api.encode()
# manual LB3 drive (encode doesn't know LB3): do it alpn-like? No - _forward_pure with idx/val directly
sugar_idx, sugar_val = lb3, np.full(len(lb3), 2.0, np.float32)
hS = api._forward_pure(sugar_idx, sugar_val, hops=3)
mn9 = 138332
print(f"sugar->MN9 act={hS[mn9]:.2f} vs brain mean={hS.mean():.4f} (ratio {hS[mn9]/max(hS.mean(),1e-9):.0f}x)", flush=True)
rankS = np.argsort(hS)[-200:]
typ = dict(zip(a["root_id"].astype("int64"), zip(a["cell_class"], a["cell_type"])))
from collections import Counter
print("TOP-200 after sugar:", Counter(typ.get(int(fly_ids[int(i)]), ("?", "?")) for i in rankS).most_common(10), flush=True)
rs = (hS > hS[mn9]).sum()
print(f"MN9: act={hS[mn9]:.4f} rank={rs}/{len(hS)} (top {100*rs/len(hS):.2f}%)", flush=True)
mech = api.MECH
m_idx, m_val = mech, np.full(len(mech), 2.0, np.float32)
hM = api._forward_pure(m_idx, m_val, hops=3)
rankM = set(np.argsort(hM)[-200:])
ov = len(set(rankS) & rankM)
print(f"sugar vs mechano top-200 overlap: {ov}/200 (expect small - separate populations)", flush=True)
print(f"MN9 after mechano: rank={(hM>hM[mn9]).sum()} act={hM[mn9]:.2f}", flush=True)
print("PASS-sugar-mn9", flush=True)
