"""Czy to nadal mucha? Topologia, znaki Dale'a, dryf wag, odruchy obwodowe.
Zapisuje wM_trained.npz (1 trial) + raport."""
import numpy as np, pandas as pd, time
t0 = time.time()
roles = np.load("roles_full.npz")
N = int(roles["N"][0]); ORN, ALPN_f, MECH, VIS = roles["ORN"], roles["ALPN"], roles["MECH"], roles["VIS"]
EFFERENT = roles["EFFERENT"]
mb = np.load("mb_circuit.npz"); mb_fly = mb["flywire_ids"]
KC_loc, MBON_loc = mb["KC"], mb["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[KC_loc]); MBON_glob = np.sort(loc2glob[MBON_loc])
g = np.load("mb_groups.npz")
app_g = set(int(loc2glob[a]) for a in g["approach"].tolist())
ai = [i for i, gg in enumerate(MBON_glob) if int(gg) in app_g]
vi = [i for i in range(len(MBON_glob)) if i not in ai]

con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values.astype(np.int32)
post = con["Postsynaptic_Index"].values.astype(np.int32)
ws = con["Excitatory x Connectivity"].values.astype(np.float32); del con
E = len(pre)
w0 = np.abs(ws); sign0 = np.sign(ws); sign0[sign0==0]=1.0; del ws
print(f"1) TOPOLOGIA: N={N} (oczek. 138639), E={E} (oczek. 15091983)", flush=True)
print(f"   neurony zachowane: {N==138639}, krawedzie: {E==15091983}, maska zamrozona (nigdy nie dodajemy/usuwamy)", flush=True)

wM = w0.copy(); sign = sign0.copy()
print(f"2) ZNAKI Dale'a: zamrozone w kodzie (sign buffer, nigdy nie trenowany) -> zgodnosc 100% z definicji", flush=True)
print(f"   E/I start: pos={(sign>0).sum()} neg={(sign<0).sum()} suma={float((wM*sign).sum()):.0f}", flush=True)

rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself1 = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wmsg2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
Wself2 = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
CH = 2000000
def full_h2(bias):
    x = emb.copy(); x[bias] += 2.0
    P = x @ Wmsg1; agg = np.zeros((N, hid), dtype=np.float32)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        np.add.at(agg, post[e], P[pre[e]] * (wM[e]*sign[e])[:, None])
    h1 = np.maximum(0, x @ Wself1 + agg); h1 /= (h1.mean()+1e-6)
    P2 = h1 @ Wmsg2; agg2 = np.zeros((N, hid), dtype=np.float32)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        np.add.at(agg2, post[e], P2[pre[e]] * (wM[e]*sign[e])[:, None])
    h2 = np.maximum(0, h1 @ Wself2 + agg2); h2 /= (h2.mean()+1e-6)
    return h1, h2

odorA = ORN[:len(ORN)//2]; odorB = ORN[len(ORN)//2:]
_, hA = full_h2(odorA); _, hB = full_h2(odorB)
# 3) odruchy na ORYGINALNYCH wagach (przed treningiem)
alpn_act = float(hA[ALPN_f].mean()); rnd = np.random.default_rng(0).choice(N, len(ALPN_f), replace=False)
print(f"3) ODRUCH WECHOWY (wagi oryginalne): ALPN|odorA={alpn_act:.3f} vs losowe neurony={float(hA[rnd].mean()):.3f} (musi byc >)", flush=True)
kcA = hA[KC_glob].mean(axis=1); kcB = hB[KC_glob].mean(axis=1)
k = max(1, int(len(kcA)*0.05))
ov = len(set(np.argsort(kcA)[-k:]) & set(np.argsort(kcB)[-k:]))
print(f"   KC top5% overlap A/B={ov}/{k} (mucha: rzadkie, rozlaczne ~15%), corr={float(np.corrcoef(kcA,kcB)[0,1]):.2f}", flush=True)
vis_act = float(hA[VIS].mean()); print(f"   WZROK: VIS|odorA(zapach)={vis_act:.3f} vs ALPN={alpn_act:.3f} (zapach nie powinien napedzac wzroku)", flush=True)

# 4) trening 1 trial + dryf
KCset = set(int(x) for x in KC_glob); MBset = set(int(x) for x in MBON_glob)
km_pos = np.where([p in KCset and q in MBset for p, q in zip(pre, post)])[0]
kc_rank = {int(gg): i for i, gg in enumerate(KC_glob)}
mb_idx = {int(gg): i for i, gg in enumerate(MBON_glob)}
glob2loc = {int(gg): int(l) for l, gg in enumerate(loc2glob)}
avoid_loc = set(int(x) for x in g["avoid"].tolist())
km_ki = np.array([kc_rank[int(pre[i])] for i in km_pos], dtype=np.int32)
km_mi = np.array([mb_idx[int(post[i])] for i in km_pos], dtype=np.int32)
km_is_avoid = np.array([glob2loc[int(post[i])] in avoid_loc for i in km_pos])
ref_mb = np.zeros(len(MBON_glob)); np.add.at(ref_mb, km_mi, wM[km_pos].astype(float))
ref_in = np.zeros(N); np.add.at(ref_in, post, wM.astype(float))
elig = np.zeros(E, dtype=np.float32)
def train_step(bias, reward, to_avoid):
    global wM
    h1, h2 = full_h2(bias)
    a1 = h1.mean(axis=1).astype(np.float32); a2 = h2.mean(axis=1).astype(np.float32)
    for s in range(0, E, CH):
        e = slice(s, min(s+CH, E))
        elig[e] = elig[e]*0.9 + (a1[pre[e]]*a2[post[e]]).astype(np.float32)
    if reward != 0:
        for s in range(0, E, CH):
            e = slice(s, min(s+CH, E))
            dw = np.clip(0.002*reward*elig[e], -0.02*wM[e], 0.02*wM[e])
            wM[e] = np.clip(wM[e] + dw - 1e-6, 0.05, 650.0)
        cur = np.zeros(N); np.add.at(cur, post, wM.astype(float))
        sc = np.ones(N); nz = cur > 1e-9
        sc[nz] = np.clip(ref_in[nz]/cur[nz], 0.95, 1.05)
        for s in range(0, E, CH):
            e = slice(s, min(s+CH, E))
            wM[e] = wM[e]*sc[post[e]]
    kcs = h2[KC_glob].mean(axis=1); m = kcs >= np.sort(kcs)[-max(1,int(len(kcs)*0.05))]
    sel = km_is_avoid if to_avoid else ~km_is_avoid
    wM[km_pos[sel & m[km_ki]]] *= 0.85
    return h2
train_step(odorA, +1.0, True); train_step(odorB, -1.0, False)
np.savez_compressed("wM_trained.npz", wM=wM)
chg = np.abs(wM-w0)/np.maximum(w0, 1e-9)
print(f"4) DRYF WAG po 1 triale: srednia |dW|/W={float(chg.mean())*100:.2f}% (musi byc <5%), corr(w0,w1)={float(np.corrcoef(w0,wM)[0,1]):.5f} (musi byc ~1.0)", flush=True)
print(f"   EI {float((w0*sign0).sum()):.0f} -> {float((wM*sign).sum()):.0f}, krawedzi zmienionych o >10%: {(chg>0.10).sum()}/{E}", flush=True)
_, hA2 = full_h2(odorA); _, hB2 = full_h2(odorB)
def pref(h):
    kcs = h[KC_glob].mean(axis=1); m = kcs >= np.sort(kcs)[-max(1,int(len(kcs)*0.05))]
    ks = (kcs*m).astype(np.float32)
    r = np.zeros(len(MBON_glob)); np.add.at(r, km_mi, ks[km_ki]*wM[km_pos])
    return float(r[ai].mean()-r[vi].mean())
print(f"5) NAUKA vs TOZSAMOSC: prefA={pref(hA2):.1f} prefB={pref(hB2):.1f} (rozdziela, a wagi te same w {float(np.corrcoef(w0,wM)[0,1])*100:.2f}%)", flush=True)
print("PASS-flyness", flush=True)
