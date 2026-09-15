"""Replikacja: konflikt ucieczka-vs-karmienie (paper SNN v783, grudzien 2025).
Protokol papieru: loom = drive LC4/LPLC2 @150Hz (faza terminalna); odczyt synergia
ucieczki (DNp01 trigger, DNp04 postawa, DNa02 ster) + MN9 (CB0701) i pula premotoryczna
(top driverzy MN9 z konektomu - 'Roundup' nie wystepuje w anotacji supp1 v783).
Dawka: 25/50/100% populacji LC4/LPLC2. Wyciszanie: DNge031 / CB0565 / oba (additive logic).
JEDNOSTKI: rate pure (j.a.), raportujemy zmiany wzgledne %, nie Hz.
"""
import numpy as np, pandas as pd, time
t0 = time.time()
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="full", path="."); api.enable_scaling()

comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
f2i = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
typ = dict(zip(a["root_id"].astype(int), a["cell_type"]))
LC4 = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t == "LC4" and int(r) in f2i}), np.int32)
LPLC2 = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t == "LPLC2" and int(r) in f2i}), np.int32)
def pair(t):
    return np.array(sorted({f2i[int(r)] for r, tt in typ.items() if tt == t and int(r) in f2i}), np.int32)
DNp01, DNp04, DNa02, DNge031, CB0565 = pair("DNp01"), pair("DNp04"), pair("DNa02"), pair("DNge031"), pair("CB0565")
MN9 = np.array([f2i[720575940660219265]], np.int32)
con = pd.read_parquet("Connectivity_783.parquet",
    columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
dMN = con[con["Postsynaptic_Index"] == MN9[0]].sort_values("Excitatory x Connectivity", ascending=False)
PREM = dMN.head(8)["Presynaptic_Index"].values.astype(np.int32)  # pula premotoryczna (data-driven)
print(f"LC4={len(LC4)} LPLC2={len(LPLC2)} DN={len(DNp01)}/{len(DNp04)}/{len(DNa02)} "
      f"DNge031={len(DNge031)} CB0565={len(CB0565)} PREM=8 (top w={dMN.head(8)['Excitatory x Connectivity'].values})", flush=True)
del con

def drive_LC(frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n1, n2 = int(len(LC4) * frac), int(len(LPLC2) * frac)
    i1 = rng.choice(LC4, n1, replace=False) if n1 else np.zeros(0, np.int32)
    i2 = rng.choice(LPLC2, n2, replace=False) if n2 else np.zeros(0, np.int32)
    idx = np.concatenate([i1, i2]); val = np.full(len(idx), 2.0, np.float32)
    return idx, val

def full_step(extra_idx=None, extra_val=None, sugar=False):
    idx, val, _ = api.encode(odor="sugar" if sugar else None)
    if extra_idx is not None and len(extra_idx):
        idx = np.concatenate([idx, extra_idx]); val = np.concatenate([val, extra_val])
    return api._forward_pure(idx, val, hops=2, thr=0.0)

def rep(h, tag):
    e = {n: float(h[v].mean()) for n, v in
         [("DNp01", DNp01), ("DNp04", DNp04), ("DNa02", DNa02), ("MN9", MN9), ("PREM", PREM)]}
    print(f"{tag}: DNp01={e['DNp01']:.4f} DNp04={e['DNp04']:.4f} DNa02={e['DNa02']:.4f} "
          f"MN9={e['MN9']:.4f} PREM={e['PREM']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return e

b = rep(full_step(sugar=True), "sugar      ")
i0, v0 = drive_LC(1.0)
l = rep(full_step(i0, v0), "loom100%   ")
c = rep(full_step(i0, v0, sugar=True), "sugar+loom ")
sup = (b["MN9"] - c["MN9"]) / max(b["MN9"], 1e-9) * 100
supP = (b["PREM"] - c["PREM"]) / max(b["PREM"], 1e-9) * 100
print(f"SUPRESJA MN9: {sup:.1f}% (papier: ~100%, 80Hz->cisza); PREM: {supP:.1f}%", flush=True)
for f in [0.25, 0.5]:
    ii, vv = drive_LC(f, seed=1)
    e = rep(full_step(ii, vv, sugar=True), f"sugar+loom{int(f*100):3d}%")
    print(f"  dawka {int(f*100)}%: MN9={e['MN9'] / max(b['MN9'], 1e-9) * 100:.1f}% baseline'u", flush=True)
# wyciszanie: DNge031 / CB0565 / oba (additive logic)
wM0 = api.wM.copy()
for nm, who in [("silDNge031", DNge031), ("silCB0565", CB0565), ("silOBA", np.concatenate([DNge031, CB0565]))]:
    api.wM[:] = wM0
    m = np.isin(api.pre, who)
    api.wM[m] = 0.0
    e = rep(full_step(i0, v0, sugar=True), nm)
    print(f"  {nm}: MN9 recovery={e['MN9'] / max(b['MN9'], 1e-9) * 100:.1f}% baseline'u", flush=True)
api.wM[:] = wM0
print("DONE-loom", flush=True)
