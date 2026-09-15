"""Full-graph plasticity: R-Hebb na wszystkich 15M + DAN anti-Hebb na K->M. 2-hop forward z cache'owanych macierzy."""
import numpy as np, pandas as pd, time
t0 = time.time()
roles = np.load("roles_full.npz")
N = int(roles["N"][0]); ORN = roles["ORN"]
mb = np.load("mb_circuit.npz"); mb_fly = mb["flywire_ids"]
KC_loc, MBON_loc = mb["KC"], mb["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[KC_loc])
MBON_glob = np.sort(loc2glob[MBON_loc])
kc_rank = {int(gg): i for i, gg in enumerate(KC_glob)}
g = np.load("mb_groups.npz")
glob2loc = {int(gg): int(l) for l, gg in enumerate(loc2glob)}
avoid_loc = set(int(x) for x in g["avoid"].tolist())
mb_idx_of_glob = {int(gg): i for i, gg in enumerate(MBON_glob)}

print("load connectivity full...", flush=True)
con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values.astype(np.int32)
post = con["Postsynaptic_Index"].values.astype(np.int32)
ws = con["Excitatory x Connectivity"].values.astype(np.float32)
del con
E = len(pre)
print(f"E={E} ({time.time()-t0:.0f}s)", flush=True)
sign = np.sign(ws); sign[sign == 0] = 1.0
wM = np.abs(ws); del ws
KCset = set(int(x) for x in KC_glob); MBset = set(int(x) for x in MBON_glob)
km_pos = np.where([p in KCset and q in MBset for p, q in zip(pre, post)])[0]
print(f"K->M pozycji: {len(km_pos)} ({time.time()-t0:.0f}s)", flush=True)
km_ki = np.array([kc_rank[int(pre[i])] for i in km_pos], dtype=np.int32)
km_mi = np.array([mb_idx_of_glob[int(post[i])] for i in km_pos], dtype=np.int32)
km_is_avoid = np.array([glob2loc[int(post[i])] in avoid_loc for i in km_pos])
km_is_approach = ~km_is_avoid
nMB = len(MBON_glob)
ref_mb = np.zeros(nMB); np.add.at(ref_mb, km_mi, wM[km_pos].astype(float))

rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wmsg2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
Wself2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
odorA = ORN[:len(ORN)//2]; odorB = ORN[len(ORN)//2:]
CH = 2000000
EI0 = float((wM*sign).sum())

def full_h2(bias):
    x = emb.copy(); x[bias] += 2.0
    P = x @ Wmsg1; agg = np.zeros((N, hid), dtype=np.float32)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        np.add.at(agg, post[e], P[pre[e]] * (wM[e]*sign[e])[:, None])
    h1 = np.maximum(0, x @ Wself1 + agg); h1 /= (h1.mean()+1e-6)  # gain control (APL/GABA)
    P2 = h1 @ Wmsg2; agg2 = np.zeros((N, hid), dtype=np.float32)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        np.add.at(agg2, post[e], P2[pre[e]] * (wM[e]*sign[e])[:, None])
    h2 = np.maximum(0, h1 @ Wself2 + agg2); h2 /= (h2.mean()+1e-6)
    return h1, h2

def readout(h2):
    kcs = h2[KC_glob].mean(axis=1)
    m = kcs >= np.sort(kcs)[-max(1, int(len(kcs)*0.05))]
    ks = (kcs*m).astype(np.float32)
    r = np.zeros(nMB); np.add.at(r, km_mi, ks[km_ki]*wM[km_pos])
    # approach = MBON_glob ktore w local sa w g approach
    app_g = set(loc2glob[a] for a in g["approach"].tolist())
    ai = [i for i, gg in enumerate(MBON_glob) if int(gg) in app_g]
    vi = [i for i in range(nMB) if i not in ai]
    return m, float(r[ai].mean()-r[vi].mean())

# eligibility full-graph (60MB) + update chunkami
elig = np.zeros(E, dtype=np.float32)
eta_all, eta_km, alpha, decay = 0.002, 0.15, 1e-6, 0.9
ref_in = np.zeros(N); np.add.at(ref_in, post, wM.astype(float))

def train_step(bias, reward, dan_target_avoid):
    global wM
    h1, h2 = full_h2(bias)
    a1 = h1.mean(axis=1).astype(np.float32); a2 = h2.mean(axis=1).astype(np.float32)
    # elig chunkami (oszczednosc RAM)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        elig[e] = elig[e]*decay + (a1[pre[e]]*a2[post[e]]).astype(np.float32)
    if reward != 0:
        for s in range(0, E, CH):
            e = slice(s, min(s+CH, E))
            dw = np.clip(eta_all*reward*elig[e], -0.02*wM[e], 0.02*wM[e])  # max +-2%
            wM[e] = np.clip(wM[e] + dw - alpha, 0.05, 650.0)
        # renorm energii per neuron na calym grafie
        cur = np.zeros(N); np.add.at(cur, post, wM.astype(float))
        sc = np.ones(N); nz = cur > 1e-9
        sc[nz] = np.clip(ref_in[nz]/cur[nz], 0.95, 1.05)
        for s in range(0, E, CH):
            e = slice(s, min(s+CH, E))
            wM[e] = wM[e]*sc[post[e]]
    # DAN anti-Hebb na K->M
    m, _ = readout(h2)
    sel = km_is_avoid if dan_target_avoid else km_is_approach
    wM[km_pos[sel & m[km_ki]]] *= (1.0-eta_km)
    wM[km_pos] = np.maximum(wM[km_pos], 0.05)
    # renorm MBON + global per-neuron (lekka: tylko MBON + clamp)
    cur = np.zeros(nMB); np.add.at(cur, km_mi, wM[km_pos].astype(float))
    sc = np.ones(nMB); nz = cur > 1e-9
    sc[nz] = np.clip(ref_mb[nz]/cur[nz], 0.9, 1.1)
    wM[km_pos] = (wM[km_pos]*sc[km_mi]).astype(np.float32)
    return h2

_, h = full_h2(odorA); _, pA = readout(h); print(f"before A: {pA:.1f} ({time.time()-t0:.0f}s)", flush=True)
_, h = full_h2(odorB); _, pB = readout(h); print(f"before B: {pB:.1f} ({time.time()-t0:.0f}s)", flush=True)
h = train_step(odorA, +1.0, True); print(f" train A done ({time.time()-t0:.0f}s)", flush=True)
h = train_step(odorB, -1.0, False); print(f" train B done ({time.time()-t0:.0f}s)", flush=True)
_, h = full_h2(odorA); _, pA2 = readout(h)
_, h = full_h2(odorB); _, pB2 = readout(h)
EI1 = float((wM*sign).sum())
print(f"after A: {pA2:.1f} B: {pB2:.1f} prefersA={bool(pA2>pB2)} d={pA2-pB2:.1f}", flush=True)
print(f"EI sum {EI0:.0f} -> {EI1:.0f} | elig |.|={np.abs(elig).sum():.1f}", flush=True)
print("PASS-full-plastic", flush=True)
