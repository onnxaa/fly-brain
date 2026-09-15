"""Brian-MB z prawdziwymi wzorcami ALPN (DoOR->wagi): geosmin vs hexanone3."""
import numpy as np
from brian2 import *
pat = np.load("alpn_patterns.npz")
ALPN_glob = pat["ALPN_glob"]
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
ws = d["weight"].astype(float)
N = int(max(pre.max(), post.max())+1)
ALPN = d["inputs_ALPN"]; MBON = d["MBON"]
g = np.load("mb_groups_v3.npz")
mb_pos = {m: i for i, m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]
# ALPN_glob -> local
mb = d
import pandas as pd
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in d["flywire_ids"]], dtype=np.int32)
glob2loc = {int(gg): int(l) for l, gg in enumerate(loc2glob)}
eqs = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''
P = {"v_0": -52*mV, "v_rst": -52*mV, "v_th": -45*mV, "t_mbr": 20*ms,
     "tau": 5*ms, "t_rfc": 2.2*ms, "t_dly": 1.8*ms, "w_syn": 0.275*mV, "f_poi": 250}
def trial(name):
    rng = np.random.default_rng(7)
    v = pat[name]
    v = v/v.max()*150.0  # szczyt = 150Hz jak Shiu
    neu = NeuronGroup(N, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=pre, j=post); syn.w = ws*P["w_syn"]
    idx, times = [], []
    for gg, rate in zip(ALPN_glob, v):
        if rate < 1.0:
            continue
        loc = glob2loc[int(gg)]
        t = 0.0
        while True:
            t += rng.exponential(1000.0/rate)
            if t >= 1000.0:
                break
            idx.append(loc); times.append(round(t, 1))
    idx = np.array(idx); times = np.array(times)
    keep = np.ones(len(idx), bool); seen = set()
    for k in range(len(idx)):
        key = (idx[k], times[k])
        if key in seen:
            keep[k] = False
        seen.add(key)
    order = np.argsort(times[keep])
    sg = SpikeGeneratorGroup(N, idx[keep][order], times[keep][order]*ms)
    drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
    drv.connect(condition="i == j"); drv.wdrv = P["w_syn"]*P["f_poi"]
    mon = SpikeMonitor(neu)
    run(1000*ms)
    cnt = np.zeros(len(MBON)); mbset = {int(m): k for k, m in enumerate(MBON)}
    for k, vv in mon.spike_trains().items():
        if int(k) in mbset:
            cnt[mbset[int(k)]] += len(vv)
    return cnt
cG = trial("geosmin"); print(f"geosmin: app={cG[ai].mean():.1f} avo={cG[vi].mean():.1f}", flush=True)
cH = trial("hexanone3"); print(f"hexanone3: app={cH[ai].mean():.1f} avo={cH[vi].mean():.1f}", flush=True)
print(f"prefG={cG[ai].mean()-cG[vi].mean():+.1f} prefH={cH[ai].mean()-cH[vi].mean():+.1f}", flush=True)
print("DONE-pat-brian", flush=True)
