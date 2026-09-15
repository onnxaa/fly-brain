"""Full-brain v2: odor = ORN_half + ALPN_half (1-hop do KC), reszta jak v1."""
import numpy as np, pandas as pd, time
import pyarrow.parquet as pq
t0 = time.time()
roles = np.load("roles_full.npz")
N = int(roles["N"][0])
ORN, ALPN_f, EFFERENT = roles["ORN"], roles["ALPN"], roles["EFFERENT"]
mb = np.load("mb_circuit.npz"); mb_fly = mb["flywire_ids"]
KC_loc, MBON_loc = mb["KC"], mb["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[KC_loc]); MBON_glob = np.sort(loc2glob[MBON_loc])
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
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
pf = pq.ParquetFile("Connectivity_783.parquet")
eff_sorted = np.sort(EFFERENT); eff_A, eff_B = eff_sorted[:len(eff_sorted)//2], eff_sorted[len(eff_sorted)//2:]
oA_rn, oB_rn = ORN[:len(ORN)//2], ORN[len(ORN)//2:]
oA_pn, oB_pn = ALPN_f[:len(ALPN_f)//2], ALPN_f[len(ALPN_f)//2:]
odorA = np.concatenate([oA_rn, oA_pn]); odorB = np.concatenate([oB_rn, oB_pn])
print(f"odorA={len(odorA)} (ORN{oA_rn.size}+ALPN{oA_pn.size}) odorB={len(odorB)}", flush=True)

def full_h(bias):
    x = emb.copy(); x[bias] += 2.0
    agg = np.zeros((N, hid), dtype=np.float32)
    for b in pf.iter_batches(batch_size=1000000, columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"]):
        pb = np.asarray(b["Presynaptic_Index"]).astype(np.int32)
        qb = np.asarray(b["Postsynaptic_Index"]).astype(np.int32)
        wb = np.asarray(b["Excitatory x Connectivity"]).astype(np.float32)
        np.add.at(agg, qb, (x[pb] @ Wmsg) * wb[:, None])
    return np.maximum(0, x @ Wself + agg)

def readout(h):
    kcs = h[KC_glob].mean(axis=1)
    m = kcs >= np.sort(kcs)[-max(1,int(len(kcs)*0.05))]
    kc_sp = (kcs*m).astype(np.float32)
    r = np.zeros(len(mb_sorted_loc)); np.add.at(r, km_mi, kc_sp[km_ki]*wKM)
    ai = [mb_loc_pos[a] for a in np.sort(g['approach'])]
    vi = [mb_loc_pos[a] for a in np.sort(g['avoid'])]
    return m, float(r[ai].mean()-r[vi].mean()), float(h[eff_A].mean()-h[eff_B].mean())

eta = 0.15
def depress(mask, to_avoid):
    global wKM
    sel = np.logical_and(km_is_avoid if to_avoid else km_is_approach, mask[km_ki])
    wKM[sel] *= (1.0-eta); wKM[:] = np.maximum(wKM, 0.05)
    cur = np.zeros(len(mb_sorted_loc)); np.add.at(cur, km_mi, wKM.astype(float))
    sc = np.ones(len(mb_sorted_loc)); nz = cur > 1e-9
    sc[nz] = np.clip(ref_mb[nz]/cur[nz], 0.9, 1.1)
    wKM[:] = (wKM*sc[km_mi]).astype(np.float32)

h = full_h(odorA); m, pA_mb, pA_mo = readout(h); print(f"before A: MB={pA_mb:.1f} motor={pA_mo:.4f} ({time.time()-t0:.0f}s)", flush=True)
h = full_h(odorB); m, pB_mb, pB_mo = readout(h); print(f"before B: MB={pB_mb:.1f} motor={pB_mo:.4f} ({time.time()-t0:.0f}s)", flush=True)
for t in range(2):
    h = full_h(odorA); mA, _, _ = readout(h); depress(mA, True)
    h = full_h(odorB); mB, _, _ = readout(h); depress(mB, False)
    print(f" trial {t+1} ({time.time()-t0:.0f}s)", flush=True)
h = full_h(odorA); _, pA_mb, pA_mo = readout(h)
h = full_h(odorB); _, pB_mb, pB_mo = readout(h)
print(f"after A: MB={pA_mb:.1f} motor={pA_mo:.4f} | B: MB={pB_mb:.1f} motor={pB_mo:.4f}", flush=True)
print(f"MB prefersA={bool(pA_mb>pB_mb)} d={pA_mb-pB_mb:.1f}", flush=True)
print("PASS-full-v2", flush=True)
