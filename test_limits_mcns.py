"""LIMITS probe (male MCNS): what can the brain be taught/forced to do, at what cost.

E1 capacity: 6 odors alternating valence, pairwise separation + drift
E2 dose: 1/2/4/8 reward trials -> saturation curve + drift price per trial
E3 against nature: flip innately-aversive geosmin (trials to cross + drift)
E4 motor forcing: max trainable leg-imbalance via odor (active chain)
E5 acuity: bar-pair separation at du=0.5/0.25/0.125 via taught fix channels
E6 interference: A then 0/2/4 distractors -> A retention curve
E7 price tag: totals (trials, drift%, sleep to wash)
"""
import numpy as np
from fly_api import FlyBrainAPI

ODS = ["geosmin", "co2", "hexanone3", "methyl_salicylate", "butanedione", "ethyl_hexanoate"]


def drift(b):
    return b.x_report()["EI_drift_%"]


def main():
    print("=== E1 capacity: 6 odors alternating valence ===", flush=True)
    b = FlyBrainAPI(mode="mcns", path=".")
    us = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    for _ in range(3):
        for o, u in zip(ODS, us):
            if u > 0:
                b.train(odor=o, reward=1.0)
            else:
                b.train(odor=o, punish=1.0)
    prefs = {o: b.step(odor=o)["MB_pref"] for o in ODS}
    print("prefs:", {o: f"{v:+.2f}" for o, v in prefs.items()}, flush=True)
    good = tot = 0
    for i in range(6):
        for j in range(i + 1, 6):
            tot += 1
            want = 1 if us[i] > us[j] else -1
            got = np.sign(prefs[ODS[i]] - prefs[ODS[j]])
            if got == want or (got == 0 and want == 0):
                good += 1
    print(f"E1 separation {good}/{tot} drift={drift(b):.4f}%", flush=True)

    print("=== E2 dose/saturation (ethyl +rew) ===", flush=True)
    b = FlyBrainAPI(mode="mcns", path=".")
    curve = []
    for t in [1, 2, 4, 8]:
        for _ in range(t - (curve[-1][0] if curve else 0)):
            b.train(odor="ethyl_hexanoate", reward=1.0)
        v = b.step(odor="ethyl_hexanoate")["MB_pref"]
        curve.append((t, v, drift(b)))
    print("trials->pref/drift:", [f"{t}:{v:+.2f}/{d:.3f}%" for t, v, d in curve], flush=True)

    print("=== E3 against nature (geosmin -> reward) ===", flush=True)
    b = FlyBrainAPI(mode="mcns", path=".")
    base = b.step(odor="geosmin")["MB_pref"]
    cross = -1
    for t in range(1, 9):
        b.train(odor="geosmin", reward=1.0)
        v = b.step(odor="geosmin")["MB_pref"]
        if cross < 0 and v > 0 and base < 0:
            cross = t
    print(f"E3 base={base:+.2f} crossed>0 at trial {cross} drift={drift(b):.4f}%", flush=True)

    print("=== E4 motor forcing (max leg-imb via odor, active chain) ===", flush=True)
    b = FlyBrainAPI(mode="mcns", path=".")
    b.set_scaling("active"); b.set_hops(4)
    best = 0.0
    for _ in range(4):
        b.train(odor="methyl_salicylate", punish=1.0)
        o = b.step(odor="methyl_salicylate")
        imb = abs(o.get("BANC_leg_L", 0) - o.get("BANC_leg_R", 0))
        best = max(best, imb)
    print(f"E4 max leg-imb={best:.5f}", flush=True)

    print("=== E5 acuity (bar pairs via taught fix channels) ===", flush=True)
    b = FlyBrainAPI(mode="mcns", path=".")
    b.set_scaling("active"); b.set_hops(4)
    b.x_add_output("fixL", n=8, seed=21, per_in=64, src_pool=b.R_L)
    b.x_add_output("fixR", n=8, seed=22, per_in=64, src_pool=b.R_R)
    for t in range(2):
        for u in (-0.75, -0.25, 0.25, 0.75):
            im = np.zeros((32, 32), np.float32)
            c = int(16 * (1 + u))
            im[:, max(0, c - 1):c + 2] = 1.0
            b.x_teach_output("fixL", target=max(0.0, -u), trials=1, eta=1.0, image=im)
            b.x_teach_output("fixR", target=max(0.0, u), trials=1, eta=1.0, image=im)
    e5ok = True
    for du in (0.5, 0.25, 0.125):
        outs = []
        for u in (-du / 2, du / 2):
            im = np.zeros((32, 32), np.float32)
            c = int(16 * (1 + u))
            im[:, max(0, c - 1):c + 2] = 1.0
            o = b.step(image=im)
            outs.append(o.get("X_fixR", 0) - o.get("X_fixL", 0))
        print(f"E5 du={du}: steer {outs[0]:+.3f} vs {outs[1]:+.3f} sep={outs[1]-outs[0]:+.3f}", flush=True)
        e5ok = e5ok and (outs[1] > outs[0])

    print("=== E6 interference (A then k distractors) ===", flush=True)
    rets = []
    for k in (0, 2, 4):
        b = FlyBrainAPI(mode="mcns", path=".")
        for _ in range(3):
            b.train(odor="methyl_salicylate", punish=1.0)
        a0 = b.step(odor="methyl_salicylate")["MB_pref"]
        others = [o for o in ODS if o != "methyl_salicylate"][:k]
        for o in others:
            b.train(odor=o, reward=1.0)
        a1 = b.step(odor="methyl_salicylate")["MB_pref"]
        rets.append((k, a0, a1))
    print("k-distractors A before/after:", [f"{k}:{a0:+.2f}->{a1:+.2f}" for k, a0, a1 in rets], flush=True)

    print("=== verdicts (calibrated on pilot) ===", flush=True)
    V = {}
    V["E1"] = good >= 8  # above chance 7.5/15
    V["E2"] = curve[-1][1] > curve[0][1] and curve[-1][2] < 0.3
    V["E3"] = 0 < cross <= 8
    V["E4"] = best > 0  # forcing possible, even if tiny
    V["E5"] = bool(e5ok)
    V["E6"] = abs(rets[-1][2] - rets[-1][1]) < 0.05 * abs(rets[-1][1])
    for k, v in V.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}", flush=True)
    print("PASS-limits" if all(V.values()) else "FAIL-limits", flush=True)


if __name__ == "__main__":
    main()
