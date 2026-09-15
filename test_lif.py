"""LIF data-only v2: port rownan Shiu/Brian na numpy (bez Briana).
Fix wzgledem v1 (test_lif_old.py):
 1) znaki Dale'a zachowane (signed w, nie abs) - 11k hamujacych w MB + APL ze znakiem z danych
 2) jednostki jak w Brianie: g-synapsa (mV), w_syn=0.275mV, tau_syn=5ms, t_mbr=20ms,
    v0=v_rst=-52mV, vth=-45mV (STALY prog; v1 mial EL+Q*fan co przy fanie MBON~3056
    dawalo Vth~-9mV i martwy MBON)
 3) refrakcja 2ms + reset do v0 (v1: Vr=-65 > EL=-70, zero refrakcji -> ALPN 900Hz)
 4) delay aksonalny 2ms (bufor, jak t_dly=1.8ms w Brianie; v1: instant)
 5) drive przez g wagami Briana (wdrv=w_syn*250, nie skok v*M_DRIVE*fan)
 6) APL jako 2 kolcowe neurony z PRAWDZIWYMI wagami KC->APL / APL->KC
    (v1: wolny integrator graded tau=500ms -> za slaby, KC 13.8%).
    JEDYNY parametr fizjologii: APL_VTH_FACTOR=4 (prog APL 4x wyzej: -24mV).
    Uzasadnienie: APL to olbrzymi neuron (~3000 wejsc KC, ogromna powierzchnia
    blony -> niskie R_we -> potrzeba ~x wiecej pradu do tej samej depolaryzacji;
    bez tego pojedyncze olbrzymie synapsy K->APL (max 499) zapalaja APL do 200Hz
    i dusza KC do 0% / MBON do ciszy). Dobrany do rzadkosci KC~5%
    (kryterium Shiu: KC<5Hz i 5-10%), nie do konkretnych liczb Briana.
 7) PROTOKOL: drive z prawdziwych wzorcow DoOR (alpn_patterns.npz, R=150 jak
    assay_fit.py), nie polowki ALPN @150Hz (51300Hz lacznie vs ~2000Hz real -
    25x za mocny, kazdy LIF - Brian tez: MBON 57Hz - eksploduje).
Stale wylacznie z protokolu Briana/Shiu - zero strojenia pod fan.
"""
import numpy as np, time
from collections import deque
t0 = time.time()

d = np.load("mb_circuit.npz")
pre, post = d["pre"].astype(np.int32), d["post"].astype(np.int32)
ws = d["weight"].astype(np.float64)  # SIGNED
ALPN, KC, MBON = d["inputs_ALPN"], d["KC"], d["MBON"]
N = int(max(pre.max(), post.max()) + 1)

# --- APL: prawdziwe wagi z konektomu (jak assay_apl.py) ---
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

# --- rozszerzony graf N+2 (2x APL kolcowe, te same parametry LIF) ---
nN = N + 2
pre2 = np.concatenate([pre, k2a_pre, N + a2k_apl])
post2 = np.concatenate([post, N + k2a_apl, a2k_post])
w2 = np.concatenate([ws, k2a_w, -a2k_w])  # APL->KC hamujace (dane: Exc=-1)

# --- drive z prawdziwych wzorcow DoOR (jak assay_fit.py) ---
pat = np.load("alpn_patterns.npz")
AG = pat["ALPN_glob"]

g = np.load("mb_groups_v3.npz")
mb_pos = {m: i for i, m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]

# --- parametry Shiu/Brian (protokol, nie strojenie) ---
V0, VRST, VTH = -52.0, -52.0, -45.0   # mV
T_MBR, TAU_SYN = 20.0, 5.0            # ms
DT, T = 1.0, 1000                     # ms, krokow
REFR_STEPS, DELAY_STEPS = 2, 2        # 2.2ms / 1.8ms
W_SYN = 0.275                         # mV na jednostke wagi
F_POI = 250.0
WDRV = W_SYN * F_POI                  # 68.75 mV na spike drive'u (jak w Brianie)
APL_VTH_FACTOR = 4.0                  # prog APL: V0+4*(VTH-V0) = -24mV (patrz pkt 6)
DEC_G = float(np.exp(-DT / TAU_SYN))
K_MBR = DT / T_MBR


def run_odor(odor="butanedione", R=150.0, seed=7):
    """Poisson drive z alpn_patterns (v/vmax*R), 1:1 z assay_fit.py."""
    vpat = pat[odor].astype(np.float64)
    vpat = vpat / vpat.max() * R
    rate_of = {}
    for gg, rate in zip(AG, vpat):
        if int(gg) in _glob2loc and rate >= 1.0:
            rate_of[_glob2loc[int(gg)]] = rate * DT / 1000.0  # p(spike)/krok
    print(f"{odor}: ALPN napedzane={len(rate_of)}/685", flush=True)
    drive_idx = np.array(sorted(rate_of), dtype=np.int32)
    drive_p = np.array([rate_of[i] for i in drive_idx], dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = np.full(nN, V0, np.float64)
    vth = np.full(nN, VTH, np.float64)
    vth[N:] = V0 + APL_VTH_FACTOR * (VTH - V0)  # APL: wysoki prog (duzy neuron)
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
