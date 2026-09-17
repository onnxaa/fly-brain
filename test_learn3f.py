"""Third-factor (DAN-gated) KC->MBON plasticity (Handler 2019/Hige 2015).

reward/punish drive PAM/PPL (dan_rew/dan_pun US pathway); measured DAN
activity GATES the update (s = clip(DAN/DAN_ref), self-calibrated).
No DAN firing -> no learning. Real FlyWire KC->MBON synapses (wM),
read out via real MBON (MB_pref). Distinguishes gated (graded + blocked)
from scripted (flat x0.85 regardless).
"""
import numpy as np
from fly_api import FlyBrainAPI

fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (" " + str(info) if info else ""), flush=True)
    if not cond:
        fails.append(name)


def pref_after(rw, block=False, n=3, pu=0.0):
    b = FlyBrainAPI(mode="mb", path=".", seed=1)
    b0 = b.step(odor="A")["MB_pref"]
    for _ in range(n):
        b.train(odor="A", reward=rw, punish=pu, dan_block=block)
    return b0, b.step(odor="A")["MB_pref"]


# US pathway separation
b = FlyBrainAPI(mode="mb", path=".", seed=1)
o = b.step(dan_rew=1.0); p = b.step(dan_pun=1.0)
check("us-sep", o.get("DAN_pam", 0) > 1.0 and p.get("DAN_ppl", 0) > 1.0
      and o.get("DAN_ppl", 9) < 0.1 and p.get("DAN_pam", 9) < 0.1,
      f"PAM={o.get('DAN_pam', -1):.2f} PPL={p.get('DAN_ppl', -1):.2f}")

# graded acquisition via real MBON
d0 = []
for rw in (0.0, 0.5, 1.0):
    a0, a1 = pref_after(rw)
    d0.append(a1 - a0)
check("graded", abs(d0[0]) < 1e-9 and 0 < d0[1] < d0[2], f"d={['%+.1f' % d for d in d0]}")

# DAN block: US without DAN -> zero learning (causal)
a0, a1 = pref_after(1.0, block=True)
check("blocked", abs(a1 - a0) < 1e-9, f"d={a1 - a0:+.4f}")

# reversal via punishment (PPL1 -> weaken approach)
b = FlyBrainAPI(mode="mb", path=".", seed=1)
b.train(odor="A", reward=1.0); b.train(odor="A", reward=1.0)
pre = b.step(odor="A")["MB_pref"]
b.train(odor="A", punish=1.0); b.train(odor="A", punish=1.0)
post = b.step(odor="A")["MB_pref"]
check("reversal", post < pre, f"{pre:+.2f} -> {post:+.2f}")

# sleep tag: learned synapses survive SHY wash (95% vs 59% untagged)
b = FlyBrainAPI(mode="mb", path=".", seed=1)
for _ in range(3):
    b.train(odor="A", reward=1.0)
pre = b.step(odor="A")["MB_pref"]
b.sleep(5, rate=0.1)
post = b.step(odor="A")["MB_pref"]
check("tag", post / pre > 0.9 if pre != 0 else False, f"{pre:+.2f} -> {post:+.2f}")

# sleep replay: consolidation ENHANCES (replay > no-replay retention)
def _rep(rp):
    bb = FlyBrainAPI(mode="mb", path=".", seed=1)
    for _ in range(3):
        bb.train(odor="A", reward=1.0)
    pp = bb.step(odor="A")["MB_pref"]
    bb.sleep(5, rate=0.1, replay=rp)
    return pp, bb.step(odor="A")["MB_pref"]


p0, q0 = _rep(0.0)
p5, q5 = _rep(0.5)
check("replay", q5 / p5 > q0 / p0 and q5 / p5 > 1.0,
      f"plain {q0 / p0:.0%} vs replay {q5 / p5:.0%}")

# R-STDP in-loop gating (Florian-like): no US -> silent, US -> plastic
import numpy as _np


def _dw3(rw):
    ss = FlyBrainAPI(mode="mb", path=".", seed=1)
    ss.set_activation("spike")
    w0 = ss.wM.copy()
    ss.train(odor="A", reward=rw, stdp=True)
    return float(_np.abs(ss.wM - w0).sum())


_d0, _d1 = _dw3(0.0), _dw3(1.0)
check("rstdp", _d0 == 0.0 and _d1 > 0, f"no-US={_d0:.1f} US={_d1:.1f}")

print("ALL PASS" if not fails else f"FAILURES: {fails}", flush=True)
raise SystemExit(1 if fails else 0)
