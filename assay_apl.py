"""DEPRECATED (2026-09-15): Brian-APL stawia boxa (OOM/zawieszenie) - nie uzywac.
Nastepca: test_lif.py v2 (numpy-LIF z kolcowym APL: KC 5.1%, PASS-lif-data).
Zostawiony jako dokumentacja proby, nie do uruchamiania.
--- oryginalny opis ---
Kolcowy APL: 2 neurony stopniowane (bez kolcow) z PRAWDZIWYMI wagami KC->APL i APL->KC.
Hamowanie ciagle przez summed variable. Porownanie KC% z/bez APL."""
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
sup = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
apl_g = [fly2idx[int(r)] for r in sup[sup["cell_type"] == "APL"]["root_id"] if int(r) in fly2idx]
con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
keep = set(int(x) for x in loc2glob)
k2a = con[con["Postsynaptic_Index"].isin(apl_g) & con["Presynaptic_Index"].isin(keep)]
a2k = con[con["Presynaptic_Index"].isin(apl_g) & con["Postsynaptic_Index"].isin(keep)]
k2a_pre = np.array([glob2loc[int(p)] for p in k2a["Presynaptic_Index"].values], dtype=np.int32)
k2a_w = np.abs(k2a["Excitatory x Connectivity"].values.astype(float))
a2k_post = np.array([glob2loc[int(q)] for q in a2k["Postsynaptic_Index"].values], dtype=np.int32)
a2k_w = np.abs(a2k["Excitatory x Connectivity"].values.astype(float))
# map APL global -> 0/1
apl_id = {g: k for k, g in enumerate(apl_g)}
k2a_apl = np.array([apl_id[int(q)] for q in k2a["Postsynaptic_Index"].values], dtype=np.int32)
a2k_apl = np.array([apl_id[int(p)] for p in a2k["Presynaptic_Index"].values], dtype=np.int32)
print(f"APL: in={len(k2a_w)} out={len(a2k_w)}", flush=True)
eqs = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
Iapl : volt
rfc : second
'''
# uwaga: Iapl dodamy jako summed; rownanie v uzyjemy z Iapl:
eqs = '''
dv/dt = (v_0 - v + g + Iapl) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
Iapl : volt
rfc : second
'''
P = {"v_0": -52*mV, "v_rst": -52*mV, "v_th": -45*mV, "t_mbr": 20*ms,
     "tau": 5*ms, "t_rfc": 2.2*ms, "t_dly": 1.8*ms, "w_syn": 0.275*mV, "f_poi": 250}
eqA = '''dV/dt = -V / tauA : volt'''
rng = np.random.default_rng(7)
v = pat["geosmin"]; v = v/v.max()*150.0
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
keepm = np.ones(len(idx), bool); seen = set()
for k in range(len(idx)):
    key = (idx[k], times[k])
    if key in seen:
        keepm[k] = False
    seen.add(key)
order = np.argsort(times[keepm])
DIDX, DTIM = idx[keepm][order], times[keepm][order]
def run(use_apl):
    if use_apl:
        E0 = len(pre)
        pre2 = np.concatenate([pre, k2a_pre])
        post2 = np.concatenate([post, N+k2a_apl])
        wB = np.concatenate([ws, k2a_w])
        pre2 = np.concatenate([pre2, N+a2k_apl])
        post2 = np.concatenate([post2, a2k_post])
        wB = np.concatenate([wB, -a2k_w])  # APL hamujacy (dane: Exc=-1)
        nN, pA, pO, wW = N+2, pre2, post2, wB
    else:
        nN, pA, pO, wW = N, pre, post, ws
    neu = NeuronGroup(nN, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]; neu.Iapl = 0*mV
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=pA, j=pO); syn.w = wW*P["w_syn"]
    sg = SpikeGeneratorGroup(N, DIDX, DTIM*ms)
    drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
    drv.connect(condition="i == j"); drv.wdrv = P["w_syn"]*P["f_poi"]
    mon = SpikeMonitor(neu)
    run(1000*ms)
    c = {int(k): len(vv) for k, vv in mon.spike_trains().items()}
    kc = np.array([c.get(int(k), 0) for k in KC], float)
    mb = np.array([c.get(int(m), 0) for m in MBON], float)
    ap = np.array([c.get(N, 0), c.get(N+1, 0)]) if use_apl else np.zeros(2)
    print(f"  KC={kc.mean():.2f}Hz akt={(kc>0).mean()*100:.1f}% MBON={mb.mean():.1f}Hz APL={ap}", flush=True)
print("bez APL:", flush=True)
run(False)
print("z APL:", flush=True)
run(True)
print("DONE-apl", flush=True)
