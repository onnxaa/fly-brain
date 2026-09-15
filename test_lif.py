"""LIF data-only v2: Shiu/Brian equations ported to numpy (no Brian).
Fixes vs v1 (test_lif_old.py):
 1) Dale signs kept (signed w, not abs) - 11k inhibitory in MB + APL with data signs
 2) Brian units: g-synapse (mV), w_syn=0.275mV, tau_syn=5ms, t_mbr=20ms,
    v0=v_rst=-52mV, vth=-45mV (FIXED threshold; v1 had EL+Q*fan which at MBON fan~3056
    gave Vth~-9mV and a dead MBON)
 3) 2ms refractory + reset to v0 (v1: Vr=-65 > EL=-70, zero refractory -> ALPN 900Hz)
 4) 2ms axonal delay (buffer, like t_dly=1.8ms in Brian; v1: instant)
 5) drive through g with Brian weights (wdrv=w_syn*250, not v jumps of M_DRIVE*fan)
 6) APL as 2 spiking neurons with REAL KC->APL / APL->KC weights
    (v1: slow graded integrator tau=500ms -> too weak, KC 13.8%).
    The ONLY physiology parameter: APL_VTH_FACTOR=4 (APL threshold 4x higher: -24mV).
    Rationale: APL is a giant neuron (~3000 KC inputs, huge membrane area
    -> low input resistance -> needs ~x more current for the same depolarization;
    without it, single giant K->APL synapses (max 499) drive APL to 200Hz
    and silence KC to 0% / MBON to quiet). Fit to KC sparsity ~5%
    (Shiu criterion: KC<5Hz and 5-10%), not to specific Brian numbers.
 7) PROTOCOL: drive from real DoOR patterns (alpn_patterns.npz, R=150 like
    assay_fit.py), not ALPN halves @150Hz (51300Hz total vs ~2000Hz real -
    25x too strong, every LIF - Brian included: MBON 57Hz - explodes).
Constants only from the Brian/Shiu protocol - zero fan tuning.
"""
import numpy as np, time
from collections import deque
t0 = time.time()

d = np.load("mb_circuit.npz")
pre, post = d["pre"].astype(np.int32), d["post"].astype(np.int32)
ws = d["weight"].astype(np.float64)  # SIGNED
ALPN, KC, MBON = d["inputs_ALPN"], d["KC"], d["MBON"]
N = int(max(pre.max(), post.max()) + 1)

# --- APL: real connectome weights (like assay_apl.py) ---
import pandas as _pd
_comp = _pd.read_csv("Completeness_783.csv")
_fly = _comp[_comp.columns[0]].values.astype("int64")
_f2i = {int(f): i for i, f in enumerate(_fly)}
_sup = _pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
_apl_g = [_f2i[int(r)] for r in _sup[_sup["cell_type"] == "APL"]["root_id"] if int(r) in _f2i]
_con = _pd.read_parquet("Connectivity_783.parquet",
    columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
_loc2glob = np.array([_f2i[int(f)] for f in d["flywire_ids"]], dtype=np.int32)
_glob2loc = {int(gg): int(ll) for ll, gg in enumerate(_loc2glob)}
_keep2loc = _glob2loc
_apl_id = {g: k for k, g in enumerate(_apl_g)}
_k2a = _con[_con["Postsynaptic_Index"].isin(_apl_g) & _con["Presynaptic_Index"].isin(_keep2loc)]
_a2k = _con[_con["Presynaptic_Index"].isin(_apl_g) & _con["Postsynaptic_Index"].isin(_keep2loc)]
k2a_pre = np.array([_keep2loc[int(p)] for p in _k2a["Presynaptic_Index"].values], dtype=np.int32)
k2a_w = np.abs(_k2a["Excitatory x Connectivity"].values.astype(np.float64))
k2a_apl = np.array([_apl_id[int(q)] for q in _k2a["Postsynaptic_Index"].values], dtype=np.int32)
a2k_post = np.array([_keep2loc[int(q)] for q in _a2k["Postsynaptic_Index"].values], dtype=np.int32)
a2k_w = np.abs(_a2k["Excitatory x Connectivity"].values.astype(np.float64))
a2k_apl = np.array([_apl_id[int(p)] for p in _a2k["Presynaptic_Index"].values], dtype=np.int32)
print(f"APL: in={len(k2a_w)} out={len(a2k_w)}", flush=True)
del _con, _comp, _sup

# --- extended N+2 graph (2x spiking APL, same LIF params) ---
nN = N + 2
pre2 = np.concatenate([pre, k2a_pre, N + a2k_apl])
post2 = np.concatenate([post, N + k2a_apl, a2k_post])
w2 = np.concatenate([ws, k2a_w, -a2k_w])  # APL->KC inhibitory (data: Exc=-1)

# --- drive from real DoOR patterns (like assay_fit.py) ---
pat = np.load("alpn_patterns.npz")
AG = pat["ALPN_glob"]

g = np.load("mb_groups_v3.npz")
mb_pos = {m: i for i, m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]

# --- Shiu/Brian params (protocol, not tuning) ---
V0, VRST, VTH = -52.0, -52.0, -45.0   # mV
T_MBR, TAU_SYN = 20.0, 5.0            # ms
DT, T = 1.0, 1000                     # ms, steps
REFR_STEPS, DELAY_STEPS = 2, 2        # 2.2ms / 1.8ms
W_SYN = 0.275                         # mV na jednostke wagi
F_POI = 250.0
WDRV = W_SYN * F_POI                  # 68.75 mV na spike drive'u (jak w Brianie)
APL_VTH_FACTOR = 4.0                  # APL threshold: V0+4*(VTH-V0) = -24mV (see point 6)
DEC_G = float(np.exp(-DT / TAU_SYN))
K_MBR = DT / T_MBR


def run_odor(odor="butanedione", R=150.0, seed=7):
    """Poisson drive z alpn_patterns (v/vmax*R), 1:1 z assay_fit.py."""
    vpat = pat[odor].astype(np.float64)
    vpat = vpat / vpat.max() * R
    rate_of = {}
    for gg, rate in zip(AG, vpat):
        if int(gg) in _glob2loc and rate >= 1.0:
            rate_of[_glob2loc[int(gg)]] = rate * DT / 1000.0  # p(spike)/step
    print(f"{odor}: ALPN driven={len(rate_of)}/685", flush=True)
    drive_idx = np.array(sorted(rate_of), dtype=np.int32)
    drive_p = np.array([rate_of[i] for i in drive_idx], dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = np.full(nN, V0, np.float64)
    vth = np.full(nN, VTH, np.float64)
    vth[N:] = V0 + APL_VTH_FACTOR * (VTH - V0)  # APL: high threshold (large neuron)
    gg = np.zeros(nN, np.float64)
    refr = np.zeros(nN, np.int16)
    n = np.zeros(nN, np.int32)
    dq = deque([np.zeros(nN, np.float64) for _ in range(DELAY_STEPS)])
    for _ in range(T):
        gg *= DEC_G
        gg += dq.popleft()
        dq.append(np.zeros(nN, np.float64))
        if len(drive_idx):
            fire = rng.random(len(drive_idx)) < drive_p
            if fire.any():
                gg[drive_idx[fire]] += WDRV
        awake = refr == 0
        v[awake] += K_MBR * (V0 - v[awake] + gg[awake])
        refr[refr > 0] -= 1
        sp = np.where(awake & (v >= vth))[0]
        if len(sp):
            v[sp] = VRST
            gg[sp] = 0.0  # jak w Brianie: reset="v = v_rst; g = 0*mV"
            refr[sp] = REFR_STEPS
            n[sp] += 1
            m = np.isin(pre2, sp)
            if m.any():
                dq[-1] += np.bincount(post2[m], weights=(w2[m] * W_SYN).astype(np.float64),
                                      minlength=nN)
    return n / T * 1000.0


rG = run_odor("geosmin"); print(f"geosmin done ({time.time()-t0:.0f}s)", flush=True)
rB = run_odor("butanedione"); print(f"butanedione done ({time.time()-t0:.0f}s)", flush=True)
for nm, r in (("geosmin", rG), ("butanedione", rB)):
    print(f"{nm}: ALPN={r[ALPN].mean():.1f}Hz KC={r[KC].mean():.2f}Hz "
          f"akt={(r[KC] > 0).mean()*100:.1f}% app={r[MBON][ai].mean():.2f}Hz "
          f"avo={r[MBON][vi].mean():.2f}Hz APL={r[N]:.1f}/{r[N+1]:.1f}Hz", flush=True)
pG = rG[MBON][ai].mean() - rG[MBON][vi].mean()
pB = rB[MBON][ai].mean() - rB[MBON][vi].mean()
print(f"spike-pref geosmin={pG:+.2f} butanedione={pB:+.2f} d={pG-pB:+.2f}", flush=True)
ok = (rB[KC] > 0).mean() > 0.01 and abs(pG - pB) > 0.01 and rB[ALPN].mean() < 100.0
print("PASS-lif-data" if ok else "FAIL-lif-data", flush=True)
