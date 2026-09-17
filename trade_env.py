"""Fly trades the market (closed-loop associative conditioning).

MEASURED (SPY+QQQ daily 2018-2024, walk-forward, 2bp costs, mb mode):
  base config -> learned helplessness (always FLAT, x0.97, hit <1%):
  noise dominates, punishment kills approach and nothing overcomes it.
  punish x0.5 -> x1.11 SPY / x1.10 QQQ, Sharpe +0.3.
  excess-return + relative rule -> x1.29/+0.46 SPY, x1.62/+0.63 QQQ
  (random x1.30/x1.89, buy-hold x2.19/x3.19). Profile: flat-to-up every
  year (0.96-1.12x), sidesteps the bear year (yr4: fly x1.04 vs BH x0.85)
  but never captures bull runs. Low-beta timer, not an alpha machine.
  Deterministic (identical across PYTHONHASHSEED).

Setup (honest classical conditioning, no numeracy/gradient planning):
- 1 step = 1 trading day. State = 6-dim market feature vector, encoded as
  an odor-vector (mb mode, ALPN halves) built from past closes ONLY.
- Action: LONG (MB_app > MB_avo, approach) vs FLAT (avoid) - no shorting.
- Reward: realized P&L of yesterday's position -> DAN 3rd factor
  (train with reward/punish); costs 2bp per flip. Fully causal:
  decide at close t from feat_t, P&L from ret_{t+1}, train on outcome.
- Baselines: buy-and-hold, seeded random, always-flat. Metrics: total
  return, Sharpe, max drawdown, hit rate. Diagonal: what the fly exploits
  (if anything) is short-horizon conditioning + reversal on regimes.
"""
import csv
import numpy as np
from fly_api import FlyBrainAPI

COST = 0.0002


def load(sym):
    cl = []
    with open(f"mkt_{sym}.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            cl.append(float(row["close"]))
    return np.array(cl)


def features(cl, t):
    """6 features from closes[..t] (no lookahead). Returns 0..1 vector."""
    c = cl[:t + 1]
    r1 = c[-1] / c[-2] - 1 if len(c) > 1 else 0.0
    r5 = c[-1] / c[-6] - 1 if len(c) > 5 else r1
    r20 = c[-1] / c[-21] - 1 if len(c) > 20 else r5
    w = c[-21:] if len(c) > 20 else c
    lr = np.diff(np.log(w))
    vol = float(lr.std() * np.sqrt(252)) if len(lr) > 2 else 0.2
    ma20 = w.mean()
    mr = c[-1] / ma20 - 1
    acc = r5 - r20
    z = lambda x, s: float(np.clip(0.5 + x / s, 0.0, 1.0))
    return np.array([z(r1, 0.04), z(r5, 0.08), z(r20, 0.15),
                     z(vol - 0.2, 0.4), z(mr, 0.1), z(acc, 0.08)],
                    dtype=np.float32)


def run_market(sym, leak=0.0, seed=1, verbose=True, punish_scale=1.0,
               excess=False, rel_rule=False):
    cl = load(sym)
    rets = cl[1:] / cl[:-1] - 1
    b = FlyBrainAPI(mode="mb", path=".", seed=seed)
    if leak > 0:
        b.set_state(leak)
    # warmup: odor baseline not needed; start after 25 bars of history
    t0 = 25
    pos = 0
    eq = 1.0
    eqs = []
    rets_fly = []
    prev_feat = None
    for t in range(t0, len(cl) - 1):
        feat = features(cl, t)
        o = b.step(odor=feat)
        if rel_rule:
            # trade relative preference (running median threshold) so mass
            # depression cannot pin the fly flat forever
            _hist = getattr(b, "_pref_hist", [])
            _pref = o["MB_app"] - o["MB_avo"]
            _thr = float(np.median(_hist)) if len(_hist) >= 20 else 0.0
            new_pos = 1 if _pref > _thr else 0
            _hist.append(_pref)
            b._pref_hist = _hist[-500:]
        else:
            new_pos = 1 if o["MB_app"] > o["MB_avo"] else 0
        if new_pos != pos:
            eq *= (1 - COST)
        pos = new_pos
        r = rets[t]  # close t -> close t+1
        pnl = pos * r
        eq *= (1 + pnl)
        eqs.append(eq)
        rets_fly.append(pnl)
        # online update on yesterday's outcome (causal: feat_{t} decided pos,
        # outcome r realized; teach that association)
        ref = rets[t] if excess else 0.0
        ex = pnl - pos * ref  # vs market (excess) or vs zero
        s = min(abs(ex) / 0.02, 1.0)
        if ex > 0:
            b.train(odor=feat, reward=s)
        elif ex < 0:
            b.train(odor=feat, punish=s * punish_scale)
        prev_feat = feat
    rets_fly = np.array(rets_fly)
    bh = cl[-1] / cl[t0]
    tot = eq
    sh = float(rets_fly.mean() / (rets_fly.std() + 1e-12) * np.sqrt(252))
    run = np.maximum.accumulate(eqs)
    dd = float(((run - eqs) / run).max())
    hit = float((rets_fly > 0).mean())
    if verbose:
        print(f"{sym} leak={leak}: fly x{tot:.2f}  BH x{bh:.2f}  Sharpe {sh:+.2f}  "
              f"maxDD {dd:.1%}  hit {hit:.1%}  (n={len(rets_fly)})", flush=True)
    return {"tot": tot, "bh": bh, "sharpe": sh, "dd": dd, "hit": hit, "eq": eqs}


def baselines(sym):
    cl = load(sym)
    t0 = 25
    rets = cl[1:] / cl[:-1] - 1
    seg = rets[t0:]
    rng = np.random.default_rng(7)
    rnd_pos = rng.integers(0, 2, len(seg))
    eq = 1.0
    flips = np.abs(np.diff(rnd_pos, prepend=0)).sum()
    eq_rnd = float(np.prod(1 + rnd_pos * seg) * (1 - COST) ** flips)
    print(f"{sym} random: x{eq_rnd:.2f}  flat: x1.00", flush=True)


if __name__ == "__main__":
    import sys
    for sym in (sys.argv[1:] or ["spy", "qqq"]):
        baselines(sym)
        for leak in (0.0, 0.3):
            run_market(sym, leak=leak)
        print("-- punish x0.5 --", flush=True)
        run_market(sym, punish_scale=0.5)
        print("-- excess + rel --", flush=True)
        run_market(sym, excess=True, rel_rule=True)
