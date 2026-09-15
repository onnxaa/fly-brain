"""VNC test: real MANC wiring brain->DESC->VNC->motor (full mode).

PASS: mech drives VNC_desc>0.01 + VNC_motor>0, mech_L lateralizes DN_L>DN_R.
Odor->DESC is ~0 in rate mode (known deep-chain limit, documented).
"""
import numpy as np
from fly_api import FlyBrainAPI


def main():
    fly = FlyBrainAPI(mode="full", path=".")
    print(fly.enable_vnc(), flush=True)
    n = len(fly.MECH)
    o = fly.step(mech=np.ones(n, np.float32))
    print(f"mech_all: VNC_desc={o['VNC_desc_mean']:.4f} motor={o['VNC_motor']:.5f} "
          f"legL={o['VNC_leg_L']:.5f} legR={o['VNC_leg_R']:.5f} wing={o['VNC_wing']:.5f}", flush=True)
    oL = fly.step(mech_left=np.ones(len(fly.MECH_L), np.float32))
    oR = fly.step(mech_right=np.ones(len(fly.MECH_R), np.float32))
    print(f"mech_L: DN={oL.get('DN_L',0):.4f}/{oL.get('DN_R',0):.4f} "
          f"VNC_desc={oL['VNC_desc_mean']:.4f} leg={oL['VNC_leg_L']:.5f}/{oL['VNC_leg_R']:.5f}", flush=True)
    print(f"mech_R: DN={oR.get('DN_L',0):.4f}/{oR.get('DN_R',0):.4f} "
          f"VNC_desc={oR['VNC_desc_mean']:.4f} leg={oR['VNC_leg_L']:.5f}/{oR['VNC_leg_R']:.5f}", flush=True)
    ok = (o["VNC_desc_mean"] > 0.01 and o["VNC_motor"] > 0
          and oL.get("DN_L", 0) > oL.get("DN_R", 0))
    print("PASS-vnc" if ok else "FAIL-vnc", flush=True)


if __name__ == "__main__":
    main()
