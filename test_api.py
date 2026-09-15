"""Test API data-only (mb): odory protokolowe A/B, trening, separacja. Obraz tylko w full."""
import numpy as np
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="mb", path=".")
oA = api.step(odor="A"); oB = api.step(odor="B")
print(f"odorA: MB={oA['MB_pref']:+.1f} KC={oA['KC_active']} | odorB: MB={oB['MB_pref']:+.1f} d={oA['MB_pref']-oB['MB_pref']:+.1f}", flush=True)
try:
    api.step(image=np.zeros((64, 64), np.float32), odor="A")
    print("FAIL: mb+obraz powinien odmowic", flush=True)
except ValueError as e:
    print(f"mb+obraz -> ValueError OK", flush=True)
r = api.train(odor="A", reward=1.0)
print(f"po A+rew: MB={r['MB_pref']:+.1f} (przed {oA['MB_pref']:+.1f})", flush=True)
api.train(odor="B", punish=1.0)
a2 = api.step(odor="A"); b2 = api.step(odor="B")
print(f"po B+pun: A={a2['MB_pref']:+.1f} B={b2['MB_pref']:+.1f} d={a2['MB_pref']-b2['MB_pref']:+.1f}", flush=True)
api.save_weights("w_api_test.npz"); api.load_weights("w_api_test.npz")
print("save/load OK", flush=True)
print("PASS-api", flush=True)
