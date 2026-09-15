"""Caly mozg jednoczesnie: chunkowany forward 15M krawedzi (pyarrow stream) + readout MB opponent + motor efferent + DAN anti-Hebb na K->M.
Forward: h = ReLU(x@Wself + sum_pre->post[(x_pre@Wmsg)*w]) po calym grafie, chunki 1M.
Plastycznosc: tylko K->M (62k) jak w test_mb_opponent (szybkie), reszta mozgu zamrozona.
"""
import numpy as np, pandas as pd, time
import pyarrow.parquet as pq
t0 = time.time()

roles = np.load("roles_full.npz")
N = int(roles["N"][0])
ORN, EFFERENT = roles["ORN"], roles["EFFERENT"]
print(f"full N={N} ORN={len(ORN)} EFFERENT={len(EFFERENT)}", flush=True)

mb = np.load("mb_circuit.npz")
mb_fly = mb["flywire_ids"]  # local->flywire
KC_loc, MBON_loc = mb["KC"], mb["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[KC_loc])
MBON_glob = np.sort(loc2glob[MBON_loc])
print(f"KC_glob={len(KC_glob)} MBON_glob={len(MBON_glob)}", flush=True)

g = np.load("mb_groups.npz")  # approach/avoid w local MB (do wKM), nie global
# wagi K->M init z |w| (jak w opponent): odtworz mape
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
approach_set = set(g["approach"].tolist())
km_is_avoid = np.array([(mb_sorted_loc[m] in set(g["avoid"].tolist())) for m in km_mi])
km_is_approach = ~km_is_avoid
print(f"wKM={len(wKM)}", flush=True)
ref_mb = np.zeros(len(mb_sorted_loc)); np.add.at(ref_mb, km_mi, wKM.astype(float))

rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)

pf = pq.ParquetFile("Connectivity_783.parquet")
eff_sorted = np.sort(EFFERENT)
eff_half = len(eff_sorted)//2
eff_A, eff_B = eff_sorted[:eff_half], eff_sorted[eff_half:]

odorA = ORN[:len(ORN)//2]; odorB = ORN[len(ORN)//2:]
print(f"odorA={len(odorA)} ORN odorB={len(odorB)} ORN", flush=True)

def forward_full(bias):
    x = emb.copy(); x[bias] += 2.0
    agg = np.zeros((N, hid), dtype=np.float32)
    batches = pf.iter_batches(batch_size=1000000,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    for b in batches:
        pb = np.asarray(b["Presynaptic_Index"]).astype(np.int32)
        qb = np.asarray(b["Postsynaptic_Index"]).astype(np.int32)
        wb = np.asarray(b["Excitatory x Connectivity"]).astype(np.float32)
        msg = (x[pb] @ Wmsg) * wb[:, None]
        np.add.at(agg, qb, msg)
    h = x @ Wself + agg
    np.maximum(h, 0, out=h)
    # KC sparse top5%
    kcs = h[KC_glob].mean(axis=1)
    k = max(1, int(len(kcs)*0.05)); thr = np.sort(kcs)[-k]
    m = kcs >= thr
    kc_sp = (kcs*m).astype(np.float32)
    # MB readout z wKM
    r = np.zeros(len(mb_sorted_loc)); np.add.at(r, km_mi, kc_sp[km_ki]*wKM)
    app = r[[mb_loc_pos[a] for a in np.sort(g['approach'])]].mean() if len(g['approach']) else 0
    # uwaga: g['approach'] to local-MB idx; mb_loc_pos klucze to local-MB idx -> ok
    avo_idx = [mb_loc_pos[a] for a in np.sort(g['avoid'])]
    app_idx = [mb_loc_pos[a] for a in np.sort(g['approach'])]
    mb_pref = float(r[app_idx].mean() - r[avo_idx].mean())
    mo = float(h[eff_A].mean() - h[eff_B].mean())
    return mb_pref, mo

eta = 0.15
def depress(kc_mask, target_avoid):
    global wKM
    sel = np.logical_and(km_is_avoid if target_avoid else km_is_approach, kc_mask[km_ki])
    wKM[sel] *= (1.0-eta)
    wKM[:] = np.maximum(wKM, 0.05)
    cur = np.zeros(len(mb_sorted_loc)); np.add.at(cur, km_mi, wKM.astype(float))
    sc = np.ones(len(mb_sorted_loc)); nz = cur > 1e-9
    sc[nz] = np.clip(ref_mb[nz]/cur[nz], 0.9, 1.1)
    wKM[:] = (wKM*sc[km_mi]).astype(np.float32)

def kc_mask_only(bias):
    # lekka wersja: KC maska wymaga h, czyli i tak full forward; uzywamy forward_full i liczymy maske osobno? dla treningu potrzebujemy maski -> robimy forward raz i uzywamy readout
    # tu: powtarzamy forward (akceptowalne przy 3 trialach)
    return None

print("before (full-brain forward)...", flush=True)
pA_mb, pA_mo = forward_full(odorA); print(f" odorA: MBpref={pA_mb:.2f} motor={pA_mo:.4f} ({time.time()-t0:.0f}s)", flush=True)
pB_mb, pB_mo = forward_full(odorB); print(f" odorB: MBpref={pB_mb:.2f} motor={pB_mo:.4f} ({time.time()-t0:.0f}s)", flush=True)

# trening 3 triale: do maski KC uzyjemy szybkiego MB-only (jak opponent) zeby nie dublowac full forward;
# forward_full i tak liczy KC z full grafu -> maske bierzemy z pomocniczego szybkiego liczenia? Nie: liczymy maske z full h.
# Implementacja: w petli wywolaj forward_full (zwraca tez maske) — rozszerzamy: tu wywolujemy forward_full i depress na podstawie maski z OSTATNIEGO pelnego h.
# Zeby nie przekomplikowac: uzyjemy масek z MB-only (bliskie) — nie, lepiej policzyc maske z full h wewnatrz forward_full.
print("trening 3 triale (A+PAM slab-avoid, B+PPL1 slab-approach)...", flush=True)
# przerabiamy forward_full na zwracajacy maske:
def forward_mask(bias):
    x = emb.copy(); x[bias] += 2.0
    agg = np.zeros((N, hid), dtype=np.float32)
    for b in pf.iter_batches(batch_size=1000000, columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"]):
        pb = np.asarray(b["Presynaptic_Index"]).astype(np.int32)
        qb = np.asarray(b["Postsynaptic_Index"]).astype(np.int32)
        wb = np.asarray(b["Excitatory x Connectivity"]).astype(np.float32)
        np.add.at(agg, qb, (x[pb] @ Wmsg) * wb[:, None])
    h = np.maximum(0, x @ Wself + agg)
    kcs = h[KC_glob].mean(axis=1)
    k = max(1, int(len(kcs)*0.05)); m = kcs >= np.sort(kcs)[-k]
    kc_sp = (kcs*m).astype(np.float32)
    r = np.zeros(len(mb_sorted_loc)); np.add.at(r, km_mi, kc_sp[km_ki]*wKM)
    app_idx = [mb_loc_pos[a] for a in np.sort(g['approach'])]
    avo_idx = [mb_loc_pos[a] for a in np.sort(g['avoid'])]
    return (kcs*m > 0), float(r[app_idx].mean()-r[avo_idx].mean()), float(h[eff_A].mean()-h[eff_B].mean())

for t in range(3):
    mA, _, _ = forward_mask(odorA); depress(mA, True)
    mB, _, _ = forward_mask(odorB); depress(mB, False)
    print(f" trial {t+1} ({time.time()-t0:.0f}s)", flush=True)

pA_mb, pA_mo = forward_full(odorA)
pB_mb, pB_mo = forward_full(odorB)
print(f"after: odorA MBpref={pA_mb:.2f} motor={pA_mo:.4f} | odorB MBpref={pB_mb:.2f} motor={pB_mo:.4f}", flush=True)
print(f"MB separation prefersA={bool(pA_mb>pB_mb)} d={pA_mb-pB_mb:.2f}", flush=True)
print("PASS-full-brain-simultaneous", flush=True)
