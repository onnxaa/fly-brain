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

print("ALL PASS" if not fails else f"FAILURES: {fails}", flush=True)
raise SystemExit(1 if fails else 0)
