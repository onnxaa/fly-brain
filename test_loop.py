"""Tropotaksja: porownaj MB_pref dla 'lewej' vs 'prawej' anteny (conc(x+d) vs conc(x-d)), idz w wyzszy.
Miara: nachylenie pref vs conc przed/po treningu A+."""
import numpy as np
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="mb", path=".")
nA = len(api.ALPN)//2; nB = len(api.ALPN)-nA
def pref_at(conc):
    odor = np.concatenate([np.full(nA, conc), np.full(nB, 0.2)]).astype(np.float32)
    return api.step(odor=odor)["MB_pref"]
cs = [0.0, 0.5, 1.0]
p0 = [pref_at(c) for c in cs]
print(f"przed: pref={['%+.0f'%v for v in p0]} nachylenie={p0[-1]-p0[0]:+.0f}", flush=True)
def run():
    x = 0.2
    for t in range(14):
        cL = float(np.clip(1.0-abs((x+0.05)-0.8)/0.8, 0, 1))
        cR = float(np.clip(1.0-abs((x-0.05)-0.8)/0.8, 0, 1))
        pL = pref_at(cL); pR = pref_at(cR)
        x = float(np.clip(x + (0.05 if pL >= pR else -0.05), 0, 1))
    return x
x0 = run()
print(f"przed treningiem x={x0:.2f} (start 0.2, cel 0.8)", flush=True)
for _ in range(2):
    odor = np.concatenate([np.full(nA, 1.0), np.full(nB, 0.2)]).astype(np.float32)
    api.train(odor=odor, reward=1.0)
p1 = [pref_at(c) for c in cs]
print(f"po: pref={['%+.0f'%v for v in p1]} nachylenie={p1[-1]-p1[0]:+.0f}", flush=True)
x1 = run()
print(f"po treningu x={x1:.2f} (oczek. >{x0:.2f}, blizej 0.8)", flush=True)
print("PASS-tropotaxis" if x1 > x0 else "FAIL-tropotaxis", flush=True)
