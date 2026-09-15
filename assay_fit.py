"""Fit ogniowy: pattern ALPN (DoOR) @ Poisson R -> ALPN~40Hz, KC<5Hz&~5-10%, MBON 5-20Hz."""
import numpy as np, sys
from brian2 import *
R = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
pat = np.load("alpn_patterns.npz")
ALPN_glob = pat["ALPN_glob"]
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
ws = d["weight"].astype(float)
N = int(max(pre.max(), post.max())+1)
ALPN = d["inputs_ALPN"]; KC = d["KC"]; MBON = d["MBON"]
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
v = pat["geosmin"]; v = v/v.max()*R
neu = NeuronGroup(N, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                  refractory="rfc", namespace=P)
neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
syn.connect(i=pre, j=post); syn.w = ws*P["w_syn"]
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
sg = SpikeGeneratorGroup(N, idx[keep][order], times[keep][order]*ms)
drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
drv.connect(condition="i == j"); drv.wdrv = P["w_syn"]*P["f_poi"]
mon = SpikeMonitor(neu)
run(1000*ms)
c = {int(k): len(vv) for k, vv in mon.spike_trains().items()}
alp = np.mean([c.get(int(i), 0) for i in ALPN])
kcr = np.array([c.get(int(i), 0) for i in KC])
mbr = np.array([c.get(int(i), 0) for i in MBON])
print(f"R={R}: ALPN={alp:.1f}Hz KC={kcr.mean():.2f}Hz akt={(kcr>0).mean()*100:.1f}% MBON={mbr.mean():.1f}Hz", flush=True)
print("DONE-fit", flush=True)
