"""Lezje: (1) KC silent -> anosmia, (2) DAN silent -> brak nauki, (3) polowa MBON -> degradacja.
Predykcje musze: d~0 po KC-lezji; d bez zmian po 1 triale bez DAN; polowa MBON slabsza separacja."""
import numpy as np
from fly_api import FlyBrainAPI
api = FlyBrainAPI(mode="mb", path=".")
base_A = api.step(odor="A")["MB_pref"]; base_B = api.step(odor="B")["MB_pref"]
print(f"baseline: A={base_A:.1f} B={base_B:.1f} d={base_A-base_B:.1f}", flush=True)
# 1) lezja KC: wyzeruj wagi K->M (wyjscie MB odciete)
w = api.wM.copy(); api.wM[api.km_mask] = 0.0
lA = api.step(odor="A")["MB_pref"]; lB = api.step(odor="B")["MB_pref"]
print(f"KC-lezja: A={lA:.1f} B={lB:.1f} d={lA-lB:.1f} (oczek. d~0, anosmia)", flush=True)
api.wM[:] = w
# 2) lezja DAN: trening bez reward/punish nie zmienia wag
ei0 = api.step(odor="A")["EI_sum"]
api.train(odor="A", reward=0.0, punish=0.0)
ei1 = api.step(odor="A")["EI_sum"]
print(f"DAN-lezja (reward=0): EI {ei0:.0f}->{ei1:.0f} (oczek. identyczne)", flush=True)
# 3) polowa MBON odcieta
api.wM[api.km_mask] = w[api.km_mask]
mb_s = np.sort(api.MBON); cut = set(mb_s[len(mb_s)//2:])
sel = np.array([q in cut for q in api.post[api.km_mask]])
api.wM[api.km_mask] = np.where(sel, 0.0, w[api.km_mask])
hA = api.step(odor="A")["MB_pref"]; hB = api.step(odor="B")["MB_pref"]
print(f"1/2 MBON: A={hA:.1f} B={hB:.1f} d={hA-hB:.1f} (oczek. |d| mniejsze niz baseline {abs(base_A-base_B):.1f})", flush=True)
ok = abs(lA-lB) < 0.05*abs(base_A-base_B) and ei0 == ei1 and abs(hA-hB) < abs(base_A-base_B)
print("PASS-lesions" if ok else "FAIL-lesions", flush=True)
