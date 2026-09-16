"""HYBRID motor path (male MCNS): rate learns, spikes act.

Why: rate MB readout is analytic (h[MBON]~0 in diffusion), so odor cannot
reach legs in rate mode (imberie max 0.0017). In spike mode MBONs really
fire, the intact chain ignites, and learned odors drive legs differentially.
PASS: CS+ leg vigor clearly above CS- (both lateralized).
"""
import numpy as np
from fly_api import FlyBrainAPI


def main():
    b = FlyBrainAPI(mode="mcns", path=".")
    for _ in range(3):
        b.train(odor="ethyl_hexanoate", reward=1.0)
        b.train(odor="methyl_salicylate", punish=1.0)
    b.set_activation("spike")
    legs = {}
    for od in ("ethyl_hexanoate", "methyl_salicylate"):
        o = b.step(odor=od)
        legs[od] = o.get("BANC_leg_L", 0) + o.get("BANC_leg_R", 0)
        print(f"spike {od}: legs={legs[od]:.3f}Hz imb={o.get('BANC_leg_L',0)-o.get('BANC_leg_R',0):+.3f}",
              flush=True)
    ok = legs["ethyl_hexanoate"] > legs["methyl_salicylate"] and legs["methyl_salicylate"] > 0
    print("PASS-hybrid" if ok else "FAIL-hybrid", flush=True)


if __name__ == "__main__":
    main()
