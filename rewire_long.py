"""LONG rewire: geosmin<->ethyl valence flip (aversive<->attractive).
Functional (magnitudes; topology+Dale frozen - project constitution).
Blocks A/B/C of 10 trials (geo+, ethyl-) + sleep(2); snapshots wM_rewire_{A,B,C}.npz.
Usage: python3 rewire_long.py [A|B|C|stab]  (B/C resume from snapshot)
Log -> results/rewire_long.txt
"""
import numpy as np, sys, time, os
t0 = time.time()
from fly_api import FlyBrainAPI
BLK = sys.argv[1] if len(sys.argv) > 1 else "A"


def log(s):
    print(s, flush=True)
    with open("results/rewire_long.txt", "a") as f:
        f.write(s + "\n")


api = FlyBrainAPI(mode="full", path="."); api.enable_scaling()
W0 = api.wM.copy()
if BLK in ("B", "C", "stab"):
    src = {"B": "wM_rewire_A.npz", "C": "wM_rewire_B.npz", "stab": "wM_rewire_C.npz"}[BLK]
    assert os.path.exists(src), src
    api.load_weights(src)
    log(f"resume z {src}")


def stats(tag):
    pg = api.step(odor="geosmin")["MB_pref"]; pe = api.step(odor="ethyl_hexanoate")["MB_pref"]
    wS = api.wM * api.sign
    chg = np.abs(api.wM - W0) / np.maximum(W0, 1e-9)
    fl = (api.wM[api.km_mask] <= 0.0501).mean() * 100
    log(f"{tag}: d(geo-eth)={pg-pe:+.2f} (geo={pg:+.2f} eth={pe:+.2f}) "
        f"EI={float(wS.sum()):.0f} dryf={float(chg.mean())*100:.2f}% floor={fl:.1f}% ({time.time()-t0:.0f}s)")


if BLK in ("A", "B", "C"):
    stats(f"pre-{BLK}")
    for t in range(1, 11):
        api.train(odor="geosmin", reward=1.0)
        api.train(odor="ethyl_hexanoate", punish=1.0)
        if t % 5 == 0:
            stats(f"{BLK}t{t}")
    api.sleep(episodes=2)
    stats(f"{BLK}+sleep")
    p = api.save_weights(f"wM_rewire_{BLK}.npz")
    log(f"snapshot {p}")
elif BLK == "stab":
    stats("pre-stab")
    for t in range(1, 6):  # neutral trials + sleep: does the flip hold?
        api.train(odor="co2", reward=0.0, punish=0.0)
    api.sleep(episodes=2)
    stats("post-stab")
print("DONE-rewire-" + BLK, flush=True)
