"""Brian2 (rownania Shiu) na obwodzie MB 6289/548k: odorA/B Poisson 150Hz -> stopy MBON approach/avoid (v3)."""
import numpy as np
from brian2 import *
prefs.codegen.target = "auto"
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
ws = d["weight"].astype(float)  # signed Excitatory x Connectivity
N = int(max(pre.max(), post.max())+1)
ALPN = d["inputs_ALPN"]; MBON = d["MBON"]
g = np.load("mb_groups_v3.npz")
mb_pos = {m: i for i, m in enumerate(np.sort(MBON))}
ai = [mb_pos[int(a)] for a in np.sort(g["approach"])]
vi = [mb_pos[int(a)] for a in np.sort(g["avoid"])]
oA = ALPN[:len(ALPN)//2]; oB = ALPN[len(ALPN)//2:]
print(f"N={N} E={len(pre)}", flush=True)

eqs = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''
P = {"v_0": -52*mV, "v_rst": -52*mV, "v_th": -45*mV, "t_mbr": 20*ms,
     "tau": 5*ms, "t_rfc": 2.2*ms, "t_dly": 1.8*ms, "w_syn": 0.275*mV,
     "r_poi": 150*Hz, "f_poi": 250}

def trial(odor):
    rng = np.random.default_rng(1)
    neu = NeuronGroup(N, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
                      refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=pre, j=post)
    syn.w = ws*P["w_syn"]
    # Poisson drive przez SpikeGenerator (PoissonInput nie podaje w brian 2.10)
    # czasy ISI-wykladnicze, deduplikacja na siatce dt
    idx, times = [], []
    for i in odor:
        t = 0.0
        while True:
            t += rng.exponential(1000.0/150.0)
            if t >= 1000.0:
                break
            idx.append(int(i)); times.append(round(t, 1))
    idx = np.array(idx); times = np.array(times)
    keep = np.ones(len(idx), bool)
    seen = set()
    for k in range(len(idx)):
        key = (idx[k], times[k])
        if key in seen:
            keep[k] = False
        seen.add(key)
    order = np.argsort(times[keep])
    sg = SpikeGeneratorGroup(N, idx[keep][order], times[keep][order]*ms)
    drv = Synapses(sg, neu, "wdrv : volt", on_pre="g += wdrv", delay=P["t_dly"])
    drv.connect(condition="i == j")
    drv.wdrv = P["w_syn"]*P["f_poi"]
    mon = SpikeMonitor(neu)
    run(1000*ms)
    cnt = np.zeros(len(MBON))
    mbset = {int(m): k for k, m in enumerate(MBON)}
    for k, v in mon.spike_trains().items():
        if int(k) in mbset:
            cnt[mbset[int(k)]] += len(v)
    return cnt  # spiki w 1s = Hz

cA = trial(oA); print(f"odorA: app={cA[ai].mean():.1f}Hz avo={cA[vi].mean():.1f}Hz", flush=True)
cB = trial(oB); print(f"odorB: app={cB[ai].mean():.1f}Hz avo={cB[vi].mean():.1f}Hz", flush=True)
print(f"prefA={cA[ai].mean()-cA[vi].mean():+.1f} prefB={cB[ai].mean()-cB[vi].mean():+.1f}", flush=True)
print("DONE-mb-brian", flush=True)
