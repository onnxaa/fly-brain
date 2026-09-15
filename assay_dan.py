"""DAN conditioning w kolcach: geosmin+PAM -> depresja KC->avoid; hex+PPL1 -> KC->approach. v3 grupy."""
import numpy as np, pandas as pd
from brian2 import *
pat = np.load("alpn_patterns.npz"); ALPN_glob = pat["ALPN_glob"]
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
ws = d["weight"].astype(float)
N = int(max(pre.max(), post.max())+1)
ALPN = d["inputs_ALPN"]; KC = d["KC"]; MBON = d["MBON"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in d["flywire_ids"]], dtype=np.int32)
glob2loc = {int(gg): int(l) for l, gg in enumerate(loc2glob)}
g = np.load("mb_groups_v3.npz")
mb_pos = {m: i for i, m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]
isK = np.zeros(N, bool); isK[KC] = True
isM = np.zeros(N, bool); isM[MBON] = True
km = np.where(isK[pre] & isM[post])[0]
kc_rank = {int(gg): i for i, gg in enumerate(np.sort(KC))}
avoid_loc = set(int(x) for x in g["avoid"])
km_is_avoid = np.array([int(post[i]) in avoid_loc for i in km])
eqs = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''
P = {"v_0": -52*mV, "v_rst": -52*mV, "v_th": -45*mV, "t_mbr": 20*ms,
     "tau": 5*ms, "t_rfc": 2.2*ms, "t_dly": 1.8*ms, "w_syn": 0.275*mV, "f_poi": 250}
RPEAK = {"geosmin": 150.0, "hexanone3": 400.0}  # zakres jak sweep Shiu (10-200Hz+)
def drive(name):
    v = pat[name]; v = v/v.max()*RPEAK[name]
    rng = np.random.default_rng(7)
    idx, times = [], []
    for gg, rate in zip(ALPN_glob, v):
        if rate < 1.0 or int(gg) not in glob2loc:
            continue
        t = 0.0
        while True:
            t += rng.exponential(1000.0/rate)
            if t >= 1000.0:
                break
            idx.append(glob2loc[int(gg)]); times.append(round(t, 1))
    idx = np.array(idx); times = np.array(times)
    keep = np.ones(len(idx), bool); seen = set()
    for k in range(len(idx)):
        key = (idx[k], times[k])
        if key in seen:
            keep[k] = False
        seen.add(key)
    order = np.argsort(times[keep])
    return idx[keep][order], times[keep][order]
def measure(drv_idx, drv_t, wscale):
    neu = NeuronGroup(N, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=pre, j=post); syn.w = (ws*wscale)*P["w_syn"]
    sg = SpikeGeneratorGroup(N, drv_idx, drv_t*ms)
    drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
    drv.connect(condition="i == j"); drv.wdrv = P["w_syn"]*P["f_poi"]
    mon = SpikeMonitor(neu)
    run(1000*ms)
    c = {int(k): len(vv) for k, vv in mon.spike_trains().items()}
    r = np.array([c.get(int(m), 0) for m in MBON], float)
    kc = np.array([c.get(int(k), 0) for k in KC], float)
    return r, kc
wscale = np.ones(len(pre))
gi, gt = drive("geosmin"); hi, ht = drive("hexanone3")
rG, kcG = measure(gi, gt, wscale)
rH, kcH = measure(hi, ht, wscale)
def pref(r):
    return r[ai].mean()-r[vi].mean()
print(f"przed: prefG={pref(rG):+.1f} prefH={pref(rH):+.1f}", flush=True)
# parowanie: aktywne KC (>0) x grupy -> depresja 15% (jak rate)
mG = kcG > 0; mH = kcH > 0
kcs = np.sort(KC)
km_ki = np.array([int(np.searchsorted(kcs, pre[i])) for i in km])
wscale2 = wscale.copy()
wscale2[km[km_is_avoid & mG[km_ki]]] *= 0.85   # geosmin+PAM: slab avoid
wscale2[km[~km_is_avoid & mH[km_ki]]] *= 0.85  # hex+PPL1: slab approach
rG2, _ = measure(gi, gt, wscale2)
rH2, _ = measure(hi, ht, wscale2)
print(f"po: prefG={pref(rG2):+.1f} prefH={pref(rH2):+.1f} (oczek. G w gore, H w dol)", flush=True)
print("PASS-dan-spike" if (pref(rG2) > pref(rG) and pref(rH2) < pref(rH)) else "FAIL-dan-spike", flush=True)
