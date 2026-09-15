"""Bateria uczenia/pamieci A: akwizycja, reversal, dyskryminacja.
Full pure, real DoOR. Uzycie: python3 learn_battery.py A|B
Wyniki -> results/learn_battery.txt (dopisyane).
"""
import numpy as np, sys, time
t0 = time.time()
from fly_api import FlyBrainAPI
PART = sys.argv[1] if len(sys.argv) > 1 else "A"
api = FlyBrainAPI(mode="full", path="."); api.enable_scaling()
W0 = api.wM.copy(); E0 = api.elig.copy()


def log(s):
    print(s, flush=True)
    with open("results/learn_battery.txt", "a") as f:
        f.write(s + "\n")


def reset():
    api.wM[:] = W0; api.elig[:] = E0
    cur = np.zeros(len(api.MBON)); np.add.at(cur, api.km_mi, api.wM[api.km_mask].astype(float))
    api.ref_mb[:] = cur


def dAB(a, b):
    pa, pb = api.step(odor=a)["MB_pref"], api.step(odor=b)["MB_pref"]
    return pa - pb, pa, pb


if PART == "A":
    log("== E1 akwizycja ethyl+/geosmin- x6 ==")
    reset()
    for t in range(7):
        d, pa, pb = dAB("ethyl_hexanoate", "geosmin")
        log(f"E1 t{t}: d={d:+.2f} (ethyl={pa:+.2f} geo={pb:+.2f}) ({time.time()-t0:.0f}s)")
        if t < 6:
            api.train(odor="ethyl_hexanoate", reward=1.0); api.train(odor="geosmin", punish=1.0)
    log("== E2 reversal: 3x ethyl+ potem 4x ethyl-/geosmin+ ==")
    reset()
    for t in range(3):
        api.train(odor="ethyl_hexanoate", reward=1.0); api.train(odor="geosmin", punish=1.0)
    d, _, _ = dAB("ethyl_hexanoate", "geosmin"); log(f"E2 pre-rev: d={d:+.2f}")
    for t in range(1, 5):
        api.train(odor="ethyl_hexanoate", punish=1.0); api.train(odor="geosmin", reward=1.0)
        d, pa, pb = dAB("ethyl_hexanoate", "geosmin")
        log(f"E2 rev{t}: d={d:+.2f} (ethyl={pa:+.2f} geo={pb:+.2f}) ({time.time()-t0:.0f}s)")
    log("== E3 dyskryminacja trudna hexanone3+/butanedione- (corr 0.39) x5 ==")
    reset()
    for t in range(6):
        d, pa, pb = dAB("hexanone3", "butanedione")
        log(f"E3h t{t}: d={d:+.2f} ({time.time()-t0:.0f}s)")
        if t < 5:
            api.train(odor="hexanone3", reward=1.0); api.train(odor="butanedione", punish=1.0)
    log("== E3 dyskryminacja latwa geosmin+/co2- x3 ==")
    reset()
    for t in range(4):
        d, pa, pb = dAB("geosmin", "co2")
        log(f"E3e t{t}: d={d:+.2f} ({time.time()-t0:.0f}s)")
        if t < 3:
            api.train(odor="geosmin", reward=1.0); api.train(odor="co2", punish=1.0)
else:
    import itertools
    log("== E4 ekstynkcja: 3x ethyl+ potem 4x ethyl solo ==")
    reset()
    for t in range(3):
        api.train(odor="ethyl_hexanoate", reward=1.0)
    d, _, _ = dAB("ethyl_hexanoate", "geosmin"); log(f"E4 post-acq: d={d:+.2f}")
    for t in range(1, 5):
        api.train(odor="ethyl_hexanoate", reward=0.0, punish=0.0)
        d, pa, pb = dAB("ethyl_hexanoate", "geosmin")
        log(f"E4 ext{t}: d={d:+.2f} ({time.time()-t0:.0f}s)")
    log("== E5 generalizacja: 3x ethyl+, sondy 6 zapachow vs geosmin ==")
    reset()
    for t in range(3):
        api.train(odor="ethyl_hexanoate", reward=1.0)
    pat = np.load("alpn_patterns.npz")
    ce = {o: float(np.corrcoef(pat["ethyl_hexanoate"], pat[o])[0, 1]) for o in
          ["geosmin", "hexanone3", "co2", "butanedione", "methyl_salicylate", "ethyl_hexanoate"]}
    for o in ["geosmin", "hexanone3", "co2", "butanedione", "methyl_salicylate", "ethyl_hexanoate"]:
        d, pa, pb = dAB(o, "geosmin") if o != "geosmin" else (0.0, api.step(odor="geosmin")["MB_pref"], 0.0)
        log(f"E5 probe {o}: d_vs_geo={d:+.2f} corr(ethyl,{o})={ce[o]:+.2f}")
    log("== E6 mieszaniny ethyl:geosmin (stosunki, bez treningu, swiezy mozdzek) ==")
    reset()
    dd = np.load("door_odors.npz")
    for re_ in [1.0, 0.5, 0.25, 0.0]:
        ie, ve = dd["ethyl_hexanoate_idx"], (dd["ethyl_hexanoate_val"] * 2.0 * re_).astype(np.float32)
        ig, vg = dd["geosmin_idx"], (dd["geosmin_val"] * 2.0 * (1.0 - re_)).astype(np.float32)
        h = api._forward_pure(np.concatenate([ie, ig]), np.concatenate([ve, vg]), hops=2, thr=0.0)
        kcs = h[api.KC]; k = max(1, int(len(kcs) * 0.05)); m = kcs >= np.sort(kcs)[-k]
        ks = (kcs * m).astype(np.float32)
        r = np.zeros(len(api.MBON)); np.add.at(r, api.km_mi, ks[api.km_ki] * api.wM[api.km_mask])
        pos = api._mb_pos
        ai = [pos[int(x)] for x in np.sort(api.approach)]; vi = [pos[int(x)] for x in np.sort(api.avoid)]
        log(f"E6 ethyl={re_:.2f}: MB_pref={float(r[ai].mean()-r[vi].mean()):+.2f}")
    log("== E7 retencja: 3x ethyl+, 6 neutralnych triali ==")
    reset()
    for t in range(3):
        api.train(odor="ethyl_hexanoate", reward=1.0)
    d, _, _ = dAB("ethyl_hexanoate", "geosmin"); log(f"E7 post-acq: d={d:+.2f}")
    for t in range(1, 7):
        api.train(odor="geosmin", reward=0.0, punish=0.0)
        d, _, _ = dAB("ethyl_hexanoate", "geosmin")
        log(f"E7 neu{t}: d={d:+.2f} ({time.time()-t0:.0f}s)")
print("DONE-battery-" + PART, flush=True)
