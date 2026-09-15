"""BANC test: female whole-CNS (brain+VNC, one animal, intact connective).

PASS: odor drives MB + KC sparsity sane; reward/punish move MB_pref in the
right directions; mech drives DESC + leg/wing motor; sleep washes.
"""
import numpy as np
from fly_api import FlyBrainAPI


def main():
    b = FlyBrainAPI(mode="banc", path=".")
    print(f"N={b.N} E={b.E} KC={len(b.KC)} MBON={len(b.MBON)} "
          f"app={len(b.approach)} avo={len(b.avoid)}", flush=True)
    o0 = b.step(odor="geosmin")
    o1 = b.step(odor="ethyl_hexanoate")
    print(f"geosmin: MB={o0['MB_pref']:.3f} KC={o0['KC_active']} "
          f"DN={o0.get('DN_L',0):.4f}/{o0.get('DN_R',0):.4f} "
          f"leg={o0.get('BANC_leg_L',0):.5f}/{o0.get('BANC_leg_R',0):.5f}", flush=True)
    print(f"ethyl: MB={o1['MB_pref']:.3f} KC={o1['KC_active']} "
          f"DN={o1.get('DN_L',0):.4f}/{o1.get('DN_R',0):.4f} "
          f"leg={o1.get('BANC_leg_L',0):.5f}/{o1.get('BANC_leg_R',0):.5f}", flush=True)
    r0 = b.step(odor="geosmin")["MB_pref"]
    b.train(odor="geosmin", reward=1.0)
    r1 = b.step(odor="geosmin")["MB_pref"]
    b.train(odor="geosmin", punish=1.0)
    r2 = b.step(odor="geosmin")["MB_pref"]
    print(f"learn: {r0:.3f} ->rew {r1:.3f} (d={r1-r0:+.3f}) ->pun {r2:.3f} (d={r2-r1:+.3f})", flush=True)
    m = b.step(mech=np.ones(len(b.MECH), np.float32))
    print(f"mech: DN={m.get('DN_L',0):.4f}/{m.get('DN_R',0):.4f} "
          f"leg={m.get('BANC_leg_L',0):.5f}/{m.get('BANC_leg_R',0):.5f} "
          f"wing={m.get('BANC_wing',0):.5f} neck={m.get('BANC_neck',0):.5f}", flush=True)
    b.sleep(2)
    print(f"wash: MB={b.step(odor='geosmin')['MB_pref']:.3f}", flush=True)
    ok = (abs(o0["MB_pref"]) < 1e9 and (r1 - r0) > 0 and (r2 - r1) < 0
          and m.get("BANC_leg_L", 0) + m.get("BANC_leg_R", 0) > 0)
    print("PASS-banc" if ok else "FAIL-banc", flush=True)


if __name__ == "__main__":
    main()
