"""Full-brain 2-hop: ORN->ALPN->KC->MBON, 2 warstwy GNN, chunk stream 2M, precomputed projekcje.
Trening: DAN anti-Hebb na K->M (jak v2). Odor: czysty ORN (bez ALPN-cheat)."""
import numpy as np, pandas as pd, time
import pyarrow.parquet as pq
t0 = time.time()
roles = np.load("roles_full.npz")
N = int(roles["N"][0])
ORN, EFFERENT = roles["ORN"], roles["EFFERENT"]
mb = np.load("mb_circuit.npz"); mb_fly = mb["flywire_ids"]
KC_loc, MBON_loc = mb["KC"], mb["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[KC_loc])
g = np.load("mb_groups.npz")
pre_l, post_l = mb["pre"], mb["post"]
w0 = np.abs(mb["weight"].astype(np.float32))
isK = np.zeros(int(pre_l.max()+1), bool); isK[KC_loc] = True
isM = np.zeros(int(pre_l.max()+1), bool); isM[MBON_loc] = True
km = isK[pre_l] & isM[post_l]
mb_sorted_loc = np.sort(MBON_loc)
mb_loc_pos = {m: i for i, m in enumerate(mb_sorted_loc)}
kc_sorted_loc = np.sort(KC_loc)
kc_loc_pos = {kk: i for i, kk in enumerate(kc_sorted_loc)}
km_ki = np.array([kc_loc_pos[a] for a in pre_l[km]], dtype=np.int32)
km_mi = np.array([mb_loc_pos[b] for b in post_l[km]], dtype=np.int32)
wKM = w0[km].copy()
avoid_set = set(g["avoid"].tolist())
km_is_avoid = np.array([mb_sorted_loc[m] in avoid_set for m in km_mi])
km_is_approach = ~km_is_avoid
ref_mb = np.zeros(len(mb_sorted_loc)); np.add.at(ref_mb, km_mi, wKM.astype(float))
rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wmsg2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
Wself2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
pf = pq.ParquetFile("Connectivity_783.parquet")
eff_sorted = np.sort(EFFERENT); eff_A, eff_B = eff_sorted[:len(eff_sorted)//2], eff_sorted[len(eff_sorted)//2:]
odorA = ORN[:len(ORN)//2]; odorB = ORN[len(ORN)//2:]
print(f"odor czysty ORN: A={len(odorA)} B={len(odorB)}", flush=True)

def agg_pass(h_in, Wmsg):
    P = h_in @ Wmsg  # (N,hid) raz
    agg = np.zeros((N, hid), dtype=np.float32)
    for b in pf.iter_batches(batch_size=2000000, columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"]):
        pb = np.asarray(b["Presynaptic_Index"]).astype(np.int32)
        qb = np.asarray(b["Postsynaptic_Index"]).astype(np.int32)
        wb = np.asarray(b["Excitatory x Connectivity"]).astype(np.float32)
        np.add.at(agg, qb, P[pb] * wb[:, None])
    return agg

def full_h2(bias):
    x = emb.copy(); x[bias] += 2.0
    h1 = np.maximum(0, x @ Wself1 + agg_pass(x, Wmsg1))
    h2 = np.maximum(0, h1 @ Wself2 + agg_pass(h1, Wmsg2))
    return h1, h2

def readout(h2):
    kcs = h2[KC_glob].mean(axis=1)
    m = kcs >= np.sort(kcs)[-max(1,int(len(kcs)*0.05))]
    kc_sp = (kcs*m).astype(np.float32)
    r = np.zeros(len(mb_sorted_loc)); np.add.at(r, km_mi, kc_sp[km_ki]*wKM)
    ai = [mb_loc_pos[a] for a in np.sort(g['approach'])]
    vi = [mb_loc_pos[a] for a in np.sort(g['avoid'])]
    return m, float(r[ai].mean()-r[vi].mean())

eta = 0.15
def depress(mask, to_avoid):
    global wKM
    sel = np.logical_and(km_is_avoid if to_avoid else km_is_approach, mask[km_ki])
    wKM[sel] *= (1.0-eta); wKM[:] = np.maximum(wKM, 0.05)
    cur = np.zeros(len(mb_sorted_loc)); np.add.at(cur, km_mi, wKM.astype(float))
    sc = np.ones(len(mb_sorted_loc)); nz = cur > 1e-9
    sc[nz] = np.clip(ref_mb[nz]/cur[nz], 0.9, 1.1)
    wKM[:] = (wKM*sc[km_mi]).astype(np.float32)

_, h = full_h2(odorA); m, pA = readout(h); print(f"before A: MB={pA:.1f} ({time.time()-t0:.0f}s)", flush=True)
_, h = full_h2(odorB); m, pB = readout(h); print(f"before B: MB={pB:.1f} ({time.time()-t0:.0f}s)", flush=True)
for t in range(1):
    _, h = full_h2(odorA); mA, _ = readout(h); depress(mA, True)
    _, h = full_h2(odorB); mB, _ = readout(h); depress(mB, False)
    print(f" trial {t+1} ({time.time()-t0:.0f}s)", flush=True)
_, h = full_h2(odorA); _, pA2 = readout(h)
_, h = full_h2(odorB); _, pB2 = readout(h)
print(f"after A: {pA2:.1f} B: {pB2:.1f} prefersA={bool(pA2>pB2)} d={pA2-pB2:.1f}", flush=True)
print("PASS-2hop", flush=True)
