"""EB ring-attractor battery (Seelig & Jayaraman 2015; Turner-Evans 2017/2020).

Rate (banc, validated ring): cue lands near target, dark holds position,
angvel walks the bump with correct sign. MCNS (functional ring,
CX_ring_validated=0): cue + dark-hold smoke only (EXPERIMENTAL).
Spike (banc): cue-evoked bump only; autonomous dark persistence and
velocity sign in spiking mode are OPEN (need per-neuron E/I homeostasis
beyond global protocol scalars) - not asserted.
"""
import numpy as np
from fly_api import FlyBrainAPI


def cue_at(b, ranks):
    EPG = np.asarray(b.CX_EPG, np.int32)
    WO = np.asarray(b.CX_EPG_wedge, np.int32)
    cue = np.zeros(len(EPG), np.float32)
    for r_ in ranks:
        cue[np.where(WO == (r_ % 50))[0]] = 1.0
    return cue


def circ_dist(a, b, n=50):
    d = (a - b) % n
    return d - n if d > n / 2 else d


fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (" " + str(info) if info else ""), flush=True)
    if not cond:
        fails.append(name)


# ---- rate, banc ----
b = FlyBrainAPI(mode="banc", path=".")
b.set_state(0.5)
assert bool(b.CX_ring_validated), "banc ring must be validated"
cue = cue_at(b, (22, 23, 24))
o = b.step(cx_cue=cue); o = b.step(cx_cue=cue)
check("rate-cue", abs(circ_dist(o.get("CX_bump", -99), 23)) <= 6, f"bump={o.get('CX_bump')}")
d0 = [b.step().get("CX_bump", -99) for _ in range(6)]
check("rate-dark", max(abs(circ_dist(x, d0[0])) for x in d0) <= 6, f"{d0}")
for AV, sgn in ((3.0, 1), (-3.0, -1)):
    b2 = FlyBrainAPI(mode="banc", path="."); b2.set_state(0.5)
    b2.step(cx_cue=cue); b2.step(cx_cue=cue)
    tr = [b2.step(angvel=AV).get("CX_bump", -99) for _ in range(6)]
    disp = sum(circ_dist(tr[i + 1], tr[i]) for i in range(len(tr) - 1))
    check(f"rate-av{AV:+}", (disp > 0) == (sgn > 0) and abs(disp) >= 4, f"{tr}")

# ---- rate, mcns (EXPERIMENTAL smoke) ----
m = FlyBrainAPI(mode="mcns", path=".")
m.set_state(0.5)
assert not bool(m.CX_ring_validated)
mcue = cue_at(m, (22, 23, 24))
o = m.step(cx_cue=mcue); o = m.step(cx_cue=mcue)
check("mcns-cue", abs(circ_dist(o.get("CX_bump", -99), 23)) <= 8, f"bump={o.get('CX_bump')}")
md = [m.step().get("CX_bump", -99) for _ in range(4)]
# looser bound: functional (unvalidated) ring = deeper wells
check("mcns-dark", max(abs(circ_dist(x, md[0])) for x in md) <= 10, f"{md}")

# ---- spike, banc: cue only ----
s = FlyBrainAPI(mode="banc", path=".")
s.set_state(0.9); s.set_activation("spike")
s.set_cx_gain(spk=2.0, std_u=0.08, bg=2.0, plat=0.8)
s._cx_gain_spk_inh = 2.0; s._cx_spk_mask = None
o = s.step(cx_cue=cue); o = s.step(cx_cue=cue)
check("spike-cue", abs(circ_dist(o.get("CX_bump", -99), 23)) <= 6 and o.get("CX_EPG", 0) > 0,
      f"bump={o.get('CX_bump')} CX_EPG={o.get('CX_EPG', -1):.2f}")

print("ALL PASS" if not fails else f"FAILURES: {fails}", flush=True)
raise SystemExit(1 if fails else 0)
