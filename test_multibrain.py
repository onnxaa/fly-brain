"""Multi-brain experiment: ensemble + coupled + fused chimera (mb mode, fast).

Question: what happens when K brains are joined into one decision system?

Architectures (all on frozen-data mb circuits, only protocol coupling):
  E1 diverse-memory ensemble: K=3 brains, b0 reward-A, b1 punish-A,
     b2 naive. Report per-brain MB_pref(A/B) + ensemble mean/vote.
  E2 noise robustness: single vs majority-vote accuracy on noisy A/B
     vectors (20 trials x 2 noise levels).
  E3 recurrent coupling (social channel): ring where each brain gets
     alpn = gain * mean(others' MB_pref) as uniform ALPN drive next step.
     PROSTHESIS (non-data, like heatbox place_code): mb has no MECH/eyes,
     so the social signal enters via generic ALPN drive. Metric: cross-
     brain std shrinks? consensus value? gain=0 vs 0.3.
  E4 weight-fused chimera ("one brain from several"): average wM of
     trained brains into a single new brain (federated averaging).
     Does it keep both memories (intermediate) or collapse?
  E5 lesion: ensemble vote without the strongest brain.

Usage: python3 test_multibrain.py
"""
import numpy as np
from fly_api import FlyBrainAPI

K = 3
SEEDS = (1, 2, 3)


def make_brains():
    brains = []
    for s in SEEDS:
        b = FlyBrainAPI(mode="mb", seed=s, path=".")
        b.enable_scaling()
        brains.append(b)
    return brains


def pref(b, odor="A"):
    return float(b.step(odor=odor)["MB_pref"])


def E1_diverse():
    print("== E1 diverse-memory ensemble ==", flush=True)
    brains = make_brains()
    for b in brains:
        print(f"  naive: A={pref(b,'A'):+.2f} B={pref(b,'B'):+.2f}", flush=True)
    brains[0].train(odor="A", reward=1.0)
    brains[1].train(odor="A", punish=1.0)
    for i, b in enumerate(brains):
        pa, pb = pref(b, "A"), pref(b, "B")
        print(f"  b{i}: A={pa:+.2f} B={pb:+.2f} d(A-B)={pa-pb:+.2f}", flush=True)
    ens_A = float(np.mean([pref(b, "A") for b in brains]))
    ens_B = float(np.mean([pref(b, "B") for b in brains]))
    votes = [1 if pref(b, "A") > 0 else -1 for b in brains]
    print(f"  ensemble-mean: A={ens_A:+.2f} B={ens_B:+.2f} "
          f"vote(A>0)={votes} sum={sum(votes):+d}", flush=True)
    return brains


def noisy_vec(n, half, which, rng, scale):
    v = rng.normal(0, scale, size=n).astype(np.float32)
    if which == "A":
        v[:half] += 1.0
    else:
        v[half:] += 1.0
    return v


def E2_noise(brains):
    print("== E2 noise robustness: single vs majority ==", flush=True)
    n = len(brains[0].ALPN)
    half = n // 2
    for scale in (0.3, 1.0):
        T = 20
        acc_single = [0.0] * len(brains)
        rng = np.random.default_rng(11)
        # paired-difference test per brain + ensemble mean
        for i, b in enumerate(brains):
            da = [float(b.step(odor=noisy_vec(n, half, "A", rng, scale))["MB_pref"])
                  for _ in range(T)]
            db = [float(b.step(odor=noisy_vec(n, half, "B", rng, scale))["MB_pref"])
                  for _ in range(T)]
            # A-vector drives first ALPN half = same half as odor 'A' prototype,
            # so A-trials should give HIGHER pref than B-trials if the code holds
            acc_single[i] = float(np.mean([1 if a > c else 0
                                           for a, c in zip(da, db)]))
        rng = np.random.default_rng(11)
        wins = 0
        for _ in range(T):
            va = noisy_vec(n, half, "A", rng, scale)
            vb = noisy_vec(n, half, "B", rng, scale)
            ea = float(np.mean([float(b.step(odor=va)["MB_pref"]) for b in brains]))
            eb = float(np.mean([float(b.step(odor=vb)["MB_pref"]) for b in brains]))
            wins += ea > eb
        acc_ens = wins / T
        print(f"  noise={scale}: single={[f'{a:.2f}' for a in acc_single]} "
              f"ensemble={acc_ens:.2f}", flush=True)


def E3_coupled(gain=0.3, steps=12):
    print(f"== E3 recurrent coupling gain={gain} ==", flush=True)
    for g in (0.0, gain):
        brains = make_brains()
        brains[0].train(odor="A", reward=1.0)
        brains[1].train(odor="A", punish=1.0)
        n = len(brains[0].ALPN)
        traj = []
        social = [np.zeros(n, np.float32) for _ in brains]
        for t in range(steps):
            prefs = []
            for i, b in enumerate(brains):
                o = b.step(odor="A", alpn=social[i])
                prefs.append(float(o["MB_pref"]))
            traj.append(prefs)
            # next-step social: uniform ALPN drive from others' mean pref
            for i in range(len(brains)):
                others = [prefs[j] for j in range(len(brains)) if j != i]
                s = float(np.clip(np.mean(others) * g * 0.1, -0.5, 0.5))
                social[i] = np.full(n, s, np.float32)
        arr = np.array(traj)
        print(f"  gain={g}: t0={arr[0].round(2).tolist()} "
              f"tend={arr[-1].round(2).tolist()} "
              f"std0={arr[0].std():.2f} stdEnd={arr[-1].std():.2f} "
              f"meanEnd={arr[-1].mean():+.2f}", flush=True)


def E4_chimera(brains):
    print("== E4 weight-fused chimera ==", flush=True)
    w0 = brains[0].wM.copy()
    w1 = brains[1].wM.copy()
    w2 = brains[2].wM.copy()
    for name, w in (("b0+b1", (w0 + w1) / 2), ("b0+b1+b2", (w0 + w1 + w2) / 3)):
        c = FlyBrainAPI(mode="mb", seed=9, path=".")
        c.enable_scaling()
        c.wM[:] = w.astype(np.float32)
        pa, pb = pref(c, "A"), pref(c, "B")
        print(f"  chimera {name}: A={pa:+.2f} B={pb:+.2f} d={pa-pb:+.2f} "
              f"(parents b0 A={pref(brains[0],'A'):+.2f} "
              f"b1 A={pref(brains[1],'A'):+.2f} "
              f"b2 A={pref(brains[2],'A'):+.2f})", flush=True)
    # E5 lesion: vote without b0
    votes_full = [1 if pref(b, "A") > 0 else -1 for b in brains]
    votes_les = votes_full[1:]
    print(f"== E5 lesion: vote full={votes_full} sum={sum(votes_full):+d} "
          f"vs no-b0={votes_les} sum={sum(votes_les):+d}", flush=True)


def main():
    brains = E1_diverse()
    E2_noise(brains)
    E3_coupled()
    E4_chimera(brains)
    print("done", flush=True)


if __name__ == "__main__":
    main()
