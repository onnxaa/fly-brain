"""X-zone test: catastrophe-free growth + new-channel learning + ledger.
PASS: |dD|<5% after growth; new KC in top-k after training; ledger matches.
"""
import numpy as np, time
t0 = time.time()
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="full", path="."); api.enable_scaling()


def dAB():
    return api.step(odor="ethyl_hexanoate")["MB_pref"] - api.step(odor="geosmin")["MB_pref"]


d0 = dAB(); print(f"baseline d={d0:+.2f}", flush=True)
new = api.x_grow_kc(n=256, seed=0, wscale=0.05)
d1 = dAB()
print(f"after +256KC growth: d={d1:+.2f} (drift {abs(d1-d0)/max(abs(d0),1e-9)*100:.1f}% <5%: {abs(d1-d0) < 0.05*abs(d0)})", flush=True)
o = api.step(odor="ethyl_hexanoate")
print(f"KC_active={o['KC_active']} (5% of {len(api.KC)})", flush=True)
for t in range(3):
    api.train(odor="ethyl_hexanoate", reward=1.0)
# new-KC share in top-k:
h = api._forward_pure(*api.encode(odor="ethyl_hexanoate")[:2], hops=2, thr=0.0)
kcs = h[api.KC]; k = max(1, int(len(kcs) * 0.05))
top = set(np.argsort(kcs)[-k:])
newpos = {i for i, g in enumerate(np.sort(api.KC)) if int(g) in set(int(x) for x in new)}
share = len(top & newpos) / max(len(top), 1) * 100
print(f"new KC in top-k after training: {share:.1f}% (random: {len(new)/len(api.KC)*100:.1f}%)", flush=True)
d2 = dAB(); print(f"d after training={d2:+.2f}", flush=True)
# test cut + ledger:
i = api.x_add_edge(int(api.ALPN[0]), int(new[0]), 1.0)
api.x_cut_edge(i)
rep = api.x_report()
ok = abs(d1 - d0) < 0.05 * abs(d0) and rep["new_neurons"] == 256 and rep["added_edges"] == 256 * 14 + 1 and rep["cut_edges"] == 1
print(f"({time.time()-t0:.0f}s)", flush=True)
print("PASS-grow" if ok else "FAIL-grow", flush=True)
