"""LIF data-only: surowe wagi, Vth=EL+0.02*fan, drive sterowany fan-in. Zero strojonych wzmocnien."""
import numpy as np, time
t0=time.time()
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
w = np.abs(d["weight"].astype(np.float32))  # SUROWE, bez skalowania
ALPN, KC, MBON = d["inputs_ALPN"], d["KC"], d["MBON"]
N = int(max(pre.max(), post.max())+1)
fan = np.zeros(N); np.add.at(fan, post, w.astype(float))
g = np.load("mb_groups_v3.npz")
mb_pos = {m:i for i,m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]
oA = ALPN[:len(ALPN)//2]; oB = ALPN[len(ALPN)//2:]
Q, M_DRIVE = 0.02, 0.015
# APL (dane!): 2 neurony, wylacznie hamujace; integrator stopniowany z PRAWDZIWYMI wagami APL->KC
import pandas as _pd
_comp = _pd.read_csv("Completeness_783.csv")
_fly = _comp[_comp.columns[0]].values.astype("int64")
_f2i = {int(f): i for i, f in enumerate(_fly)}
_sup = _pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
_apl_g = [_f2i[int(r)] for r in _sup[_sup["cell_type"] == "APL"]["root_id"] if int(r) in _f2i]
_con = _pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
_mb0 = np.load("mb_circuit.npz")
_loc2glob = np.array([_f2i[int(f)] for f in _mb0["flywire_ids"]], dtype=np.int32)
_keep2loc = {int(gg): int(ll) for ll, gg in enumerate(_loc2glob)}
_aplE = _con[_con["Presynaptic_Index"].isin(_apl_g) & _con["Postsynaptic_Index"].isin(_keep2loc)]
_apl_post = np.array([_keep2loc[int(q)] for q in _aplE["Postsynaptic_Index"].values], dtype=np.int32)
_apl_w = np.abs(_aplE["Excitatory x Connectivity"].values.astype(np.float32))
# Wejscie APL: wagi KC->APL (dane) - ta sama jednostka co Isyn, zero nowych stalych
_k2a = _con[_con["Postsynaptic_Index"].isin(_apl_g) & _con["Presynaptic_Index"].isin(_keep2loc)]
_k2a_pre = np.array([_keep2loc[int(p)] for p in _k2a["Presynaptic_Index"].values], dtype=np.int32)
_k2a_w = np.abs(_k2a["Excitatory x Connectivity"].values.astype(np.float32))
_APL_FAN = float(_k2a_w.sum())
print(f"APL: out={_apl_w.size} in={_k2a_w.size} fan_in={_APL_FAN:.0f}", flush=True)
del _con, _comp, _sup
dt, T = 1.0, 800
tau, EL, Vr = 20.0, -70.0, -65.0
Vth = (EL + Q*fan).astype(np.float32)
CH = 200000
def run(pA, pB, seed=3):
    rng = np.random.default_rng(seed)
    v = np.full(N, EL, np.float32); sp = np.zeros(N, np.float32); n = np.zeros(N)
    v_apl = 0.0
    for t in range(T):
        ev = np.zeros(N, np.float32)
        ev[oA] = (rng.random(len(oA)) < pA).astype(np.float32)
        ev[oB] = (rng.random(len(oB)) < pB).astype(np.float32)
        v += M_DRIVE*fan*ev
        v_apl += dt/500.0*(-v_apl + (_k2a_w*sp[_k2a_pre]).sum())  # w jednostkach Isyn
        Is = np.zeros(N, np.float32)
        for s in range(0, len(pre), CH):
            e = slice(s, min(s+CH, len(pre)))
            Is[post[e]] += (w[e]*sp[pre[e]]).astype(np.float32)
        rel = v_apl/max(_APL_FAN, 1e-9)  # ulamek uwolnienia 0..1
        np.add.at(Is, _apl_post, -(rel*_apl_w).astype(np.float32))
        v += dt/tau*(-(v-EL) + Is)
        s2 = v >= Vth; v[s2] = Vr; sp = s2.astype(np.float32); n += sp
    return n/T*1000.0
rA = run(0.05, 0.01); print(f"A ({time.time()-t0:.0f}s)", flush=True)
rB = run(0.01, 0.05); print(f"B ({time.time()-t0:.0f}s)", flush=True)
for nm, r, oo in (("A", rA, oA), ("B", rB, oB)):
    print(f"{nm}: ALPN={r[oo].mean():.1f}Hz KC={r[KC].mean():.2f}Hz akt={(r[KC]>0).mean()*100:.1f}% app={r[MBON][ai].mean():.2f}Hz avo={r[MBON][vi].mean():.2f}Hz", flush=True)
pA = rA[MBON][ai].mean()-rA[MBON][vi].mean(); pB = rB[MBON][ai].mean()-rB[MBON][vi].mean()
print(f"spike-pref A={pA:+.2f} B={pB:+.2f} d={pA-pB:+.2f}", flush=True)
print("PASS-lif-data" if (rA[KC]>0).mean() > 0.01 and abs(pA-pB) > 0.01 else "FAIL-lif-data", flush=True)
