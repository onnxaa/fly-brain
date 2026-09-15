"""Opponent MB: sparse KC top5% + DAN-gated anti-Hebb na K->M.
Biologia (Huang 2024, Aso 2014):
- nagroda (PAM) + KC -> oslab KC->MBON_avoid (podejscie rosnie)
- kara (PPL1) + KC -> oslab KC->MBON_approach (unikanie rosnie)
- KC sparse ~5% (hamowanie APL / WTA)
MBON approach/avoid: heurystyka v1 parzyste/nieparzyste (docelowo po kompartmencie).
"""
import numpy as np, pandas as pd, time
t0 = time.time()

d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
w = d["weight"].astype(np.float32)
N = int(max(pre.max(), post.max()) + 1)
ALPN, KC, MBON = d["inputs_ALPN"], d["KC"], d["MBON"]
print(f"N={N} E={len(pre)} ALPN={len(ALPN)} KC={len(KC)} MBON={len(MBON)}", flush=True)

# --- DAN PAM vs PPL1: mapowanie flywire -> local ---
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
idx2fly = d["flywire_ids"]
fly2loc = {int(f): l for l, f in enumerate(idx2fly)}
a = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_class", "cell_type"])
a["flywire_id"] = a["root_id"].astype(np.int64)
dan = a[a["cell_class"] == "DAN"]
pam_types = sorted([t for t in dan["cell_type"].unique() if str(t).startswith("PAM")])
ppl_types = sorted([t for t in dan["cell_type"].unique() if str(t).startswith("PPL")])
print(f"DAN typy PAM:{len(pam_types)} PPL:{len(ppl_types)}", flush=True)
loc_of = lambda fid: fly2loc.get(int(fid), -1)
dan_pam_loc, dan_ppl_loc = [], []
for _, r in dan.iterrows():
    l = loc_of(r["flywire_id"])
    if l < 0:
        continue
    if str(r["cell_type"]).startswith("PAM"):
        dan_pam_loc.append(l)
    elif str(r["cell_type"]).startswith("PPL"):
        dan_ppl_loc.append(l)
print(f"DAN_reward(PAM)={len(dan_pam_loc)} DAN_punish(PPL)={len(dan_ppl_loc)}", flush=True)

# --- MBON approach/avoid (v1: parzyste/nieparzyste po local idx) ---
mb_sorted = np.sort(MBON)
approach = mb_sorted[0::2]
avoid = mb_sorted[1::2]
mb_pos = {m: i for i, m in enumerate(mb_sorted)}
print(f"MBON approach={len(approach)} avoid={len(avoid)}", flush=True)
np.savez("mb_groups.npz", approach=approach, avoid=avoid,
         dan_pam=np.array(dan_pam_loc), dan_ppl=np.array(dan_ppl_loc))

# --- GNN do aktywacji KC (1 warstwa, jak wczesniej) ---
rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
wM = np.abs(w)
sign = np.sign(w); sign[sign == 0] = 1.0

is_KC = np.zeros(N, bool); is_KC[KC] = True
is_MBON = np.zeros(N, bool); is_MBON[MBON] = True
km_mask = is_KC[pre] & is_MBON[post]
km_pre, km_post = pre[km_mask], post[km_mask]
wKM = wM[km_mask].copy()  # liniowe wagi readout (trenowalne)
print(f"K->M: {km_mask.sum()}", flush=True)

# mapy KC-local (0..5176) i MBON-local (0..95) dla macierzy readout
kc_loc = {g: i for i, g in enumerate(np.sort(KC))}
mb_loc = {g: i for i, g in enumerate(mb_sorted)}
km_ki = np.array([kc_loc[g] for g in km_pre], dtype=np.int32)
km_mi = np.array([mb_loc[g] for g in km_post], dtype=np.int32)
approach_mi = set(mb_loc[g] for g in approach)
avoid_mi = set(mb_loc[g] for g in avoid)
km_is_avoid = np.array([m in avoid_mi for m in km_mi])
km_is_approach = ~km_is_avoid

odorA = ALPN[:len(ALPN)//2]; odorB = ALPN[len(ALPN)//2:]

def kc_sparse(bias_idx, k_frac=0.05):
    x = emb.copy(); x[bias_idx] += 2.0
    msg = (x[pre] @ Wmsg) * (wM * sign)[:, None]
    agg = np.zeros((N, hid), dtype=np.float32)
    np.add.at(agg, post, msg)
    h = np.maximum(0, x @ Wself + agg)
    s = h[KC].mean(axis=1)  # skalar per KC
    k = max(1, int(len(s) * k_frac))
    thr = np.sort(s)[-k]
    m = s >= thr
    return (s * m).astype(np.float32), m  # wartosci, maska

def readout(kc_sp):
    # kc_sp: (5177,) sparse -> (96,) MBON
    r = np.zeros(len(mb_sorted), dtype=np.float64)
    np.add.at(r, km_mi, (kc_sp[km_ki].astype(np.float64) * wKM.astype(np.float64)))
    return r

def pref(r):
    app = np.array([r[mb_loc[g]] for g in approach]).mean()
    avo = np.array([r[mb_loc[g]] for g in avoid]).mean()
    return float(app - avo), float(app), float(avo)

sA, mA = kc_sparse(odorA); sB, mB = kc_sparse(odorB)
ov = np.logical_and(mA, mB).sum()
print(f"KC top5%: active={mA.sum()} overlap A/B={ov}/{mA.sum()} ({time.time()-t0:.1f}s)", flush=True)
rA, rB = readout(sA), readout(sB)
pA, appA, avoA = pref(rA); pB, appB, avoB = pref(rB)
print(f"before: prefA={pA:.3f} (app {appA:.1f}/avo {avoA:.1f}) prefB={pB:.3f} (app {appB:.1f}/avo {avoB:.1f})", flush=True)

# --- trening DAN-gated anti-Hebb ---
eta = 0.15
ref_mb = np.zeros(len(mb_sorted)); np.add.at(ref_mb, km_mi, wKM.astype(float))

def depress(kc_mask_globalKC, target_is_avoid):
    # kc_mask_globalKC: bool (5177,) aktywne KC; cel: avoid (nagroda) lub approach (kara)
    global wKM
    # krawedzie: pre-KC aktywny I post-MBON w grupie docelowej
    sel = np.logical_and(km_is_avoid if target_is_avoid else km_is_approach,
                         kc_mask_globalKC[km_ki])
    wKM[sel] *= (1.0 - eta)
    wKM[:] = np.maximum(wKM, 0.05)
    # renorm per MBON do ref (energia)
    cur = np.zeros(len(mb_sorted)); np.add.at(cur, km_mi, wKM.astype(float))
    scale = np.ones(len(mb_sorted)); nz = cur > 1e-9
    scale[nz] = np.clip(ref_mb[nz] / cur[nz], 0.9, 1.1)
    wKM[:] = (wKM * scale[km_mi]).astype(np.float32)

for t in range(10):
    sA, mA = kc_sparse(odorA)
    depress(mA, target_is_avoid=True)   # A + PAM (nagroda): slab A->avoid
    sB, mB = kc_sparse(odorB)
    depress(mB, target_is_avoid=False)  # B + PPL1 (kara): slab B->approach
    if (t + 1) % 2 == 0:
        sA, _ = kc_sparse(odorA); sB, _ = kc_sparse(odorB)
        pA, _, _ = pref(readout(sA)); pB, _, _ = pref(readout(sB))
        print(f"trial {t+1}: prefA={pA:.3f} prefB={pB:.3f} d={pA-pB:.3f} ({time.time()-t0:.1f}s)", flush=True)

sA, _ = kc_sparse(odorA); sB, _ = kc_sparse(odorB)
pA, appA, avoA = pref(readout(sA)); pB, appB, avoB = pref(readout(sB))
print(f"after: prefA={pA:.3f} prefB={pB:.3f} d={pA-pB:.3f} prefersA={bool(pA>pB)}", flush=True)
print("PASS" if pA > pB else "NO-SEPARATION", flush=True)
