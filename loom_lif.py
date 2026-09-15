"""Looming kolcowo BEZ full-Briana: numpy-LIF (jak test_lif v2) na CALYM mozgu.
Uzycie: python3 loom_lif.py [warunek] [Tms]
warunek: sugar | loom | conflict | all (domyslnie all)
Dynamika 1:1 z test_lif v2 (Brian/Shiu: v0/rst -52, vth -45, t_mbr 20ms, tau 5ms,
refr 2ms, delay 2ms, w_syn 0.275, g-reset, APL prog x4, wdrv 68.75).
Drive: sugar-GRN @150Hz i/lub LC4+LPLC2 @150Hz (jak w papierze SNN).
Wyniki dopisywane do results/loom_lif.txt + spike counts do loom_lif_<war>.npz.
"""
import numpy as np, pandas as pd, sys, time
from collections import deque
t0 = time.time()
COND = sys.argv[1] if len(sys.argv) > 1 else "all"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 500

con = pd.read_parquet("Connectivity_783.parquet",
    columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values.astype(np.int32)
post = con["Postsynaptic_Index"].values.astype(np.int32)
ws = con["Excitatory x Connectivity"].values.astype(np.float32)
del con
roles = np.load("roles_full.npz")
N = int(roles["N"][0])
E = len(pre)
print(f"N={N} E={E} T={T}ms cond={COND}", flush=True)

comp = pd.read_csv("Completeness_783.csv")
f2i = {int(f): i for i, f in enumerate(comp[comp.columns[0]].values.astype(np.int64))}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
typ = dict(zip(a["root_id"].astype(int), a["cell_type"]))
APL = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t == "APL" and int(r) in f2i}), np.int32)
LC = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t in ("LC4", "LPLC2") and int(r) in f2i}), np.int32)
def pair(t):
    return np.array(sorted({f2i[int(r)] for r, tt in typ.items() if tt == t and int(r) in f2i}), np.int32)
DNp01, DNp04, DNa02 = pair("DNp01"), pair("DNp04"), pair("DNa02")
DNge031, CB0565 = pair("DNge031"), pair("CB0565")
MN9 = np.array([f2i[720575940660219265]], np.int32)
tt = np.load("taste_grns.npz")
SUG = tt["sugar_idx"].astype(np.int32)
con2 = pd.read_parquet("Connectivity_783.parquet",
    columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
dMN = con2[con2["Postsynaptic_Index"] == MN9[0]].sort_values("Excitatory x Connectivity", ascending=False)
PREM = dMN.head(8)["Presynaptic_Index"].values.astype(np.int32)
del con2
print(f"APL={len(APL)} LC={len(LC)} SUG={len(SUG)} DN={len(DNp01)}/{len(DNp04)}/{len(DNa02)}", flush=True)

V0, VRST, VTH = -52.0, -52.0, -45.0
DT = 1.0
REFR_STEPS, DELAY_STEPS = 2, 2
W_SYN, WDRV = 0.275, 0.275 * 250.0
APL_F = 4.0
DEC_G = float(np.exp(-DT / 5.0))
K_MBR = DT / 20.0
RATE = 150.0 * DT / 1000.0  # p(spike)/krok


def run(drive_idx, seed=7):
    rng = np.random.default_rng(seed)
    v = np.full(N, V0, np.float64)
    vth = np.full(N, VTH, np.float64)
    vth[APL] = V0 + APL_F * (VTH - V0)
    gg = np.zeros(N, np.float64)
    refr = np.zeros(N, np.int16)
    n = np.zeros(N, np.int32)
    dq = deque([np.zeros(N, np.float64) for _ in range(DELAY_STEPS)])
    di = np.asarray(drive_idx, dtype=np.int32)
    for _ in range(T):
        gg *= DEC_G
        gg += dq.popleft()
        dq.append(np.zeros(N, np.float64))
        if len(di):
            gg[di[rng.random(len(di)) < RATE]] += WDRV
        awake = refr == 0
        v[awake] += K_MBR * (V0 - v[awake] + gg[awake])
        refr[refr > 0] -= 1
        sp = np.where(awake & (v >= vth))[0]
        if len(sp):
            v[sp] = VRST
            gg[sp] = 0.0
            refr[sp] = REFR_STEPS
            n[sp] += 1
            m = np.isin(pre, sp)
            if m.any():
                dq[-1] += np.bincount(post[m], weights=(ws[m] * W_SYN).astype(np.float64), minlength=N)
    return n / T * 1000.0


def rep(r, tag):
    out = {}
    for nm, vv in [("DNp01", DNp01), ("DNp04", DNp04), ("DNa02", DNa02),
                   ("MN9", MN9), ("PREM", PREM), ("APL", APL)]:
        out[nm] = float(r[vv].mean()) if len(vv) else 0.0
    line = (f"{tag}: DNp01={out['DNp01']:.1f}Hz DNp04={out['DNp04']:.1f}Hz DNa02={out['DNa02']:.1f}Hz "
            f"MN9={out['MN9']:.1f}Hz PREM={out['PREM']:.1f}Hz APL={out['APL']:.1f}Hz ({time.time()-t0:.0f}s)")
    print(line, flush=True)
    with open("results/loom_lif.txt", "a") as f:
        f.write(line + "\n")
    return out

conds = {"sugar": SUG, "loom": LC, "conflict": np.concatenate([SUG, LC])}
todo = list(conds) if COND == "all" else [COND]
res = {}
for c in todo:
    r = run(conds[c])
    np.savez_compressed(f"loom_lif_{c}.npz", hz=r)
    res[c] = rep(r, c)
if "sugar" in res and "conflict" in res:
    b, cf = res["sugar"]["MN9"], res["conflict"]["MN9"]
    line = f"SUPRESJA MN9 kolcowo: {(b - cf) / max(b, 1e-9) * 100:.1f}% (papier ~100%)"
    print(line, flush=True)
    with open("results/loom_lif.txt", "a") as f:
        f.write(line + "\n")
print("DONE-loom-lif", flush=True)
