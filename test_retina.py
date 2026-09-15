"""Retinotopia: lewa polowa obrazu -> lewe oko. Ruch: pasek w prawo vs lewo -> kanaly R/L."""
import numpy as np
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="full", path=".")
imgL = np.zeros((64, 64), np.float32); imgL[:, :32] = 1.0
o = api.step(image=imgL)
print(f"lewy obraz: VIS_L={o['VIS_L']:.2f} VIS_R={o['VIS_R']:.2f} (L>R oczek.)", flush=True)
# ruch: pasek przesuniety w prawo miedzy klatkami
f0 = np.zeros((64, 64), np.float32); f0[:, 10:14] = 1.0
f1 = np.zeros((64, 64), np.float32); f1[:, 14:18] = 1.0
api.step(image=f0)  # zagruntuj poprzednia klatke
mR = api.step(image=f1)["motion"]
api.step(image=f1)
mL = api.step(image=f0)["motion"]
print(f"ruch w PRAWO: R={mR['R']:.1f} L={mR['L']:.1f} (R>L oczek.)", flush=True)
print(f"ruch w LEWO: R={mL['R']:.1f} L={mL['L']:.1f} (L>R oczek.)", flush=True)
ok = o["VIS_L"] > o["VIS_R"] and mR["R"] > mR["L"] and mL["L"] > mL["R"]
print("PASS-retina" if ok else "FAIL-retina", flush=True)
