"""Retinotopy: left image half -> left eye. Motion: bar right vs left -> R/L channels."""
import numpy as np
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="full", path=".")
imgL = np.zeros((64, 64), np.float32); imgL[:, :32] = 1.0
o = api.step(image=imgL)
print(f"left image: VIS_L={o['VIS_L']:.2f} VIS_R={o['VIS_R']:.2f} (expect L>R)", flush=True)
# motion: bar shifted right between frames
f0 = np.zeros((64, 64), np.float32); f0[:, 10:14] = 1.0
f1 = np.zeros((64, 64), np.float32); f1[:, 14:18] = 1.0
api.step(image=f0)  # prime the previous frame
mR = api.step(image=f1)["motion"]
api.step(image=f1)
mL = api.step(image=f0)["motion"]
print(f"motion RIGHT: R={mR['R']:.1f} L={mR['L']:.1f} (expect R>L)", flush=True)
print(f"motion LEFT: R={mL['R']:.1f} L={mL['L']:.1f} (expect L>R)", flush=True)
ok = o["VIS_L"] > o["VIS_R"] and mR["R"] > mR["L"] and mL["L"] > mL["R"]
print("PASS-retina" if ok else "FAIL-retina", flush=True)
