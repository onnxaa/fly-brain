"""Real seeing without artificial software: photons->R cells (periphery) + connectome alone.
Mapping: R pos_x/pos_y -> nearest frame pixel (like retmap.py, zero motion computation).
R1-6/R7/R8 all present in v783 (10,582). Early vision (R/L/Mi/Tm) is GRADED in physiology,
so it runs graded here (tonic release); spiking from T4/T5 up.
Drive: Poisson rate = luminance*150Hz per R cell, 100ms blocks/frame.
Dynamics: numpy-LIF 1:1 test_lif v2 on the whole brain. Readout: LC4/LPLC2, DNp01/DNp04, MN9.
Usage: python3 see_lif.py [loom|recede|static|flash|all] [Tms/block=100] [KREL] [seed]
"""
import numpy as np, pandas as pd, re, sys, time
from collections import deque
t0 = time.time()
COND = sys.argv[1] if len(sys.argv) > 1 else "all"
TBLK = int(sys.argv[2]) if len(sys.argv) > 2 else 100
KREL = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0  # Hz/mV graded release
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7

con = pd.read_parquet("Connectivity_783.parquet",
    columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values.astype(np.int32)
post = con["Postsynaptic_Index"].values.astype(np.int32)
ws = con["Excitatory x Connectivity"].values.astype(np.float32)
del con
roles = np.load("roles_full.npz")
N = int(roles["N"][0])
print(f"N={N} E={len(pre)}", flush=True)

comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
f2i = {int(f): i for i, f in enumerate(fly_ids)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type", "pos_x", "pos_y"])
typ = dict(zip(a["root_id"].astype(int), a["cell_type"]))
rm = a[a["cell_type"].isin(["R1-6", "R7", "R8"]) & (a["root_id"].astype(int).isin(f2i))].copy()
Ridx = np.array([f2i[int(r)] for r in rm["root_id"]], dtype=np.int32)
rpx = rm["pos_x"].values.astype(float); rpy = rm["pos_y"].values.astype(float)
# ommatidial grid from positions (like retmap: qcut on axes)
rx = pd.qcut(rpx, 64, labels=False, duplicates="drop").astype(int)
ry = pd.qcut(rpy, 64, labels=False, duplicates="drop").astype(int)
print(f"R={len(Ridx)} grid {rx.max()+1}x{ry.max()+1}", flush=True)

def pair(t):
    return np.array(sorted({f2i[int(r)] for r, tt in typ.items() if tt == t and int(r) in f2i}), np.int32)
LC = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t in ("LC4", "LPLC2") and int(r) in f2i}), np.int32)
DNp01, DNp04 = pair("DNp01"), pair("DNp04")
MN9 = np.array([f2i[720575940660219265]], np.int32)

V0, VRST, VTH = -52.0, -52.0, -45.0
DT = 1.0
REFR_STEPS, DELAY_STEPS = 2, 2
W_SYN, RMAX = 0.275, 150.0
APL = np.array(sorted({f2i[int(r)] for r, t in typ.items() if t == "APL" and int(r) in f2i}), np.int32)
# GRADED (physiology: early vision is non-spiking) - Poisson release d*KREL, no threshold/reset
GPAT = re.compile(r"^(R1-6|R7|R8|Lai|L[1-5]|Dm\d+|DmDRA\d+|Mi[149]|Tm[1-49]|Tm20|C[23])$")
GSET = np.array(sorted({f2i[int(r)] for r, t in typ.items() if isinstance(t, str) and GPAT.match(t) and int(r) in f2i}), np.int32)
GIS = np.zeros(N, bool); GIS[GSET] = True
print(f"GRADED={len(GSET)} (KREL={KREL}Hz/mV)", flush=True)
DEC_G = float(np.exp(-DT / 5.0))
K_MBR = DT / 20.0


def frame_rates(frame):
    """Phototransduction only: pixel luminance -> R-cell rate. Zero motion in code."""
    lum = np.asarray(frame, dtype=np.float64).reshape(64, 64)
    return (lum[ry, rx] * RMAX * DT / 1000.0).astype(np.float64)


def disk(rad, bg=0.5):
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.full((64, 64), bg, np.float32)
    img[(yy - 32) ** 2 + (xx - 32) ** 2 <= rad * rad] = 0.0
    return img


def run(frames, seed=7):
    T = TBLK * len(frames)
    P = np.array([frame_rates(f) for f in frames])  # [block, R] p(spike)/step
    rng = np.random.default_rng(seed)
    v = np.full(N, V0, np.float64)
    vth = np.full(N, VTH, np.float64)
    vth[APL] = V0 + 4.0 * (VTH - V0)
    gg = np.zeros(N, np.float64)
    refr = np.zeros(N, np.int16)
    n = np.zeros(N, np.int32)
    nblk = np.zeros((len(frames), N), np.int32)
    dq = deque([np.zeros(N, np.float64) for _ in range(DELAY_STEPS)])
    G = GSET
    for t in range(T):
        b = min(t // TBLK, len(frames) - 1)
        gg *= DEC_G
        gg += dq.popleft()
        dq.append(np.zeros(N, np.float64))
        gg[Ridx[rng.random(len(Ridx)) < P[b]]] += W_SYN * 250.0
        awake = (refr == 0) & (~GIS)
        v[awake] += K_MBR * (V0 - v[awake] + gg[awake])
        v[GIS] += K_MBR * (V0 - v[GIS] + gg[GIS])  # graded: release, no spike
        refr[refr > 0] -= 1
        sp = np.where(awake & (v >= vth))[0]
        rel = G[rng.random(len(G)) < np.clip(v[G] - V0, 0, None) * KREL * DT / 1000.0]
        ev = np.concatenate([sp, rel]) if len(rel) else sp
        if len(sp):
            v[sp] = VRST
            gg[sp] = 0.0
            refr[sp] = REFR_STEPS
            n[sp] += 1
            nblk[b, sp] += 1
        if len(ev):
            m = np.isin(pre, ev)
            if m.any():
                dq[-1] += np.bincount(post[m], weights=(ws[m] * W_SYN).astype(np.float64), minlength=N)
    return n / T * 1000.0, nblk / TBLK * 1000.0


def rep(r, blk, tag):
    lc = float(r[LC].mean())
    per = [float(blk[b][LC].mean()) for b in range(blk.shape[0])]
    line = (f"{tag}: LCmean={lc:.2f}Hz per-block={[f'{x:.1f}' for x in per]} "
            f"DNp01={float(r[DNp01].mean()):.1f}Hz DNp04={float(r[DNp04].mean()):.1f}Hz "
            f"MN9={float(r[MN9].mean()):.1f}Hz ({time.time()-t0:.0f}s)")
    print(line, flush=True)
    with open("results/see_lif.txt", "a") as f:
        f.write(line + "\n")
    return per

conds = {"loom": [disk(r) for r in (4, 10, 16, 22, 28)],
         "recede": [disk(r) for r in (28, 22, 16, 10, 4)],
         "static": [disk(4)] * 5,
         "flash": [np.zeros((64, 64), np.float32)] * 2 + [np.ones((64, 64), np.float32)] * 3}
todo = list(conds) if COND == "all" else [COND]
out = {}
for c in todo:
    r, blk = run(conds[c], seed=SEED)
    np.savez_compressed(f"see_lif_{c}.npz", hz=r, blk=blk)
    out[c] = rep(r, blk, c)
print("DONE-see-lif", flush=True)
