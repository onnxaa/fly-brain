"""Punish-resolution: hex pref przed/po, seedy 7/8/9. Faza A: baseline -> masks.npz. Faza B: depress -> wynik."""
import numpy as np, pandas as pd, sys
from brian2 import *
PHASE = sys.argv[1] if len(sys.argv) > 1 else "A"
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
avoid_loc = set(int(x) for x in g["avoid"])
km_is_avoid = np.array([int(post[i]) in avoid_loc for i in km])
kcs = np.sort(KC)
km_ki = np.array([int(np.searchsorted(kcs, pre[i])) for i in km])
eqs = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''
P = {"v_0": -52*mV, "v_rst": -52*mV, "v_th": -45*mV, "t_mbr": 20*ms,
     "tau": 5*ms, "t_rfc": 2.2*ms, "t_dly": 1.8*ms, "w_syn": 0.275*mV, "f_poi": 250}
def drive(name, seed, rpeak):
    rng = np.random.default_rng(seed)
    v = pat[name]; v = v/v.max()*rpeak
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
def measure(di, dt_, wscale):
    neu = NeuronGroup(N, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=pre, j=post); syn.w = (ws*wscale)*P["w_syn"]
    sg = SpikeGeneratorGroup(N, di, dt_*ms)
    drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
    drv.connect(condition="i == j"); drv.wdrv = P["w_syn"]*P["f_poi"]
    mon = SpikeMonitor(neu)
    run(1000*ms)
    c = {int(k): len(vv) for k, vv in mon.spike_trains().items()}
    r = np.array([c.get(int(m), 0) for m in MBON], float)
    kc = np.array([c.get(int(k), 0) for k in KC], float)
    return r, kc
w0 = np.ones(len(pre))
if PHASE == "A":
    prefs = []
    masks = []
    for s in [7, 8, 9]:
        di, dt_ = drive("hexanone3", s, 400.0)
        r, kc = measure(di, dt_, w0)
        prefs.append(r[ai].mean()-r[vi].mean())
        masks.append(kc > 0)
    mH = np.array(masks).mean(axis=0) > 0.5  # konsensus maska
    np.savez("pun_masks.npz", mH=mH)
    print(f"baseline hex pref: {np.mean(prefs):+.2f} +- {np.std(prefs):.2f}", flush=True)
else:
    z = np.load("pun_masks.npz"); mH = z["mH"]
    w1 = w0.copy()
    w1[km[~km_is_avoid & mH[km_ki]]] *= 0.7  # mocniejsza depresja
    prefs = []
    for s in [7, 8, 9]:
        di, dt_ = drive("hexanone3", s, 400.0)
        r, _ = measure(di, dt_, w1)
        prefs.append(r[ai].mean()-r[vi].mean())
    print(f"po punish hex pref: {np.mean(prefs):+.2f} +- {np.std(prefs):.2f}", flush=True)
print("DONE-phase-" + PHASE, flush=True)
