"""Fly trades the market (closed-loop associative conditioning).

MEASURED (SPY+QQQ daily 2018-2024, walk-forward, 2bp costs, mb mode):
  base config -> learned helplessness (always FLAT, x0.97, hit <1%):
  noise dominates, punishment kills approach and nothing overcomes it.
  punish x0.5 -> x1.11 SPY / x1.10 QQQ, Sharpe +0.3.
  excess-return + relative rule -> x1.29/+0.46 SPY, x1.62/+0.63 QQQ.
  OHLC features -> x1.45/+0.62 (pattern-level changes move it; level
  changes wash out in the relative rule; rehearsal/sleep/sizing null).
  CHAMPION (core-6 on live KC slots + H6 + leak 0.3 + net-of-cost):
  FULL SAMPLE: SPY x2.78/+1.05/DD21% (BH x2.19/+0.68/DD52%),
  QQQ x3.29/+1.02/DD32% (BH x3.19/+0.82/DD55%) - BEATEN both metrics both.
  TUNE 2018-21: H6 x2.21/+1.39/DD17 (BH x1.78; H4 x2.33/+1.38 second).
  HOLDOUT 2022-24 (locked, fresh): x1.12/+0.37 (BH x1.23); warm-start
  x1.16/+0.43 - positive Sharpe, trails the bull-transition. NULLS: colonies dilute,
  sleep/replay/gate/sizing move nothing (rel-rule fixed point), DD-aversion backfires;
  SHORT suicide in secular bull. Costs 2bp (5bp: x2.12, 10bp: x1.85).
  Deterministic (identical across PYTHONHASHSEED).
  CROSS-BRAIN (2024 SPY, same protocol): mb x1.12/+1.49/DD4% vs MCNS
  (male whole-CNS, chained 3x85d) x1.12/Sharpe +1.4..+2.0/DD~4% vs BH x1.24
  - IDENTICAL behavior across substrates: the market pattern, not the
  brain extract, drives the strategy. MCNS needs no dead-slot map
  (ORN 2-hop path).

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


def load_ohlc(sym):
    import csv as _csv
    import datetime as _dt
    o, h, l, c, v, yrs = [], [], [], [], [], []
    with open(f"mkt_{sym}_ohlc.csv") as f:
        for row in _csv.DictReader(f):
            o.append(float(row["open"])); h.append(float(row["high"]))
            l.append(float(row["low"])); c.append(float(row["close"]))
            v.append(float(row["volume"]))
            yrs.append(_dt.datetime.utcfromtimestamp(int(row["date"])).year)
    return {k: np.array(x) for k, x in
            (("o", o), ("h", h), ("l", l), ("c", c), ("v", v),
             ("yr", np.array(yrs)))}


_XREF = {}


def set_xref(sym, other):
    """Second market series for cross features (e.g. QQQ/SPY ratio)."""
    _XREF[sym] = load(other)


def features2(d, t):
    """10-dim past-only features (adds range/volume/gap/streak to features)."""
    c = d["c"][:t + 1]
    base = features(d["c"], t)[:6]
    h, l, o, v = d["h"][:t + 1], d["l"][:t + 1], d["o"][:t + 1], d["v"][:t + 1]
    rng = ((h[-20:] - l[-20:]) / c[-20:]).mean() if len(c) >= 20 else 0.02
    vm = v[-20:].mean() if len(v) >= 20 else 1.0
    vz = np.log(v[-1] / vm) if vm > 0 and v[-1] > 0 else 0.0
    gap = o[-1] / c[-2] - 1 if len(c) > 1 else 0.0
    lr = np.diff(np.log(c[-21:])) if len(c) > 5 else np.array([0.0])
    streak = float(np.sign(lr[-5:]).sum() / 5) if len(lr) >= 5 else 0.0
    z = lambda x, s: float(np.clip(0.5 + x / s, 0.0, 1.0))
    out = [z(rng - 0.02, 0.03), z(vz, 1.5), z(gap, 0.02), streak * 0.5 + 0.5]
    if _XREF:
        _xc = next(iter(_XREF.values()))
        if t < len(_xc):
            _ratio = _xc[max(0, t - 20):t + 1] / d["c"][max(0, t - 20):t + 1]
            _rm = _ratio[-1] / _ratio[0] - 1 if len(_ratio) > 1 else 0.0
            out.append(z(_rm, 0.05))
    return np.concatenate([base, out]).astype(np.float32)


_LIVE = None


def live_slots(b, need=16, scan=120):
    """ALPN pool positions that actually project to KC (dead slots waste
    features: measured slots 1,2,3,7,9,10,11 have zero KC fan)."""
    global _LIVE
    if _LIVE is None:
        isK = np.zeros(b.N, bool)
        isK[b.KC] = True
        live = []
        for i in range(min(scan, len(b.ALPN))):
            a = int(b.ALPN[i])
            m = b.pre == a
            if b.wM[m & isK[b.post]].sum() > 0:
                live.append(i)
        assert len(live) >= need, live
        _LIVE = live
    return _LIVE


def place(feat, live):
    """Map feature j onto live slot live[j] (zeros elsewhere = no drive)."""
    m = np.zeros(max(live[:len(feat)]) + 1, dtype=np.float32)
    for j, f in enumerate(feat):
        m[live[j]] = f
    return m


def run_market(sym, leak=0.0, seed=1, verbose=True, punish_scale=1.0,
               excess=False, rel_rule=False, ohlc=False, conf_k=0.0,
               short=False, sleep_every=0, size=False,
               horizon=1, years=None, brain=None, mode="mb",
               days=None, sleep_big=0.0, replay_top=0):
    cl = load(sym)
    rets = cl[1:] / cl[:-1] - 1
    yrs = load_ohlc(sym)["yr"] if years else None
    b = brain or FlyBrainAPI(mode=mode, path=".", seed=seed)
    _nomap = (mode != "mb")  # mb ALPN has dead slots; CNS ORN path does not
    if leak > 0:
        b.set_state(leak)
    # warmup: odor baseline not needed; start after 25 bars of history
    t0 = 25
    pos = 0
    eq = 1.0
    eqs = []
    rets_fly = []
    prev_feat = None
    hist = []  # (feat, pos, flipcost) for horizon teaching
    _rbuf = []  # hippocampal replay buffer: (feat, reward, punish) strong only
    d = load_ohlc(sym) if ohlc else None
    _live = None
    _t1 = len(cl) - 1
    if years:
        _in = [i for i in range(t0, len(cl) - 1) if int(yrs[i]) in years]
        _t1 = 0  # iterate explicit list below
    else:
        _in = list(range(t0, len(cl) - 1))
    if days:
        _in = _in[days[0]:days[1]]
    for t in _in:
        feat = features2(d, t) if ohlc else features(cl, t)
        if _live is None and not _nomap:
            _live = live_slots(b, need=len(feat))
        _fe = feat if _nomap else place(feat, _live)
        o = b.step(odor=_fe)
        if rel_rule or conf_k > 0 or short:
            # relative preference + confidence gate + optional SHORT side
            # (engineered extension: strong avoidance = active short)
            _hist = getattr(b, "_pref_hist", [])
            _pref = o["MB_app"] - o["MB_avo"]
            _thr = float(np.median(_hist)) if len(_hist) >= 20 else 0.0
            _sd = float(np.std(_hist)) if len(_hist) >= 20 else 1.0
            _hist.append(_pref)
            b._pref_hist = _hist[-500:]
            if size and not short and conf_k == 0.0:
                # fractional confidence sizing (no leverage, |dpos| costs)
                new_pos = float(np.clip((_pref - _thr) / (2 * _sd + 1e-9), 0.0, 1.0))
            elif short and _pref < _thr - conf_k * _sd:
                new_pos = -1
            elif _pref > _thr + conf_k * _sd:
                new_pos = 1
            else:
                new_pos = 0
        else:
            new_pos = 1 if o["MB_app"] > o["MB_avo"] else 0
        _fc = COST * abs(new_pos - pos) if abs(new_pos - pos) > 1e-9 else 0.0
        if _fc > 0:
            eq *= (1 - _fc)
        pos = new_pos
        r = rets[t]  # close t -> close t+1
        pnl = pos * r
        eq *= (1 + pnl)
        eqs.append(eq)
        rets_fly.append(pnl)
        # online update on yesterday's outcome (causal: feat_{t} decided pos,
        # outcome r realized; teach that association)
        ref = rets[t] if excess else 0.0
        ex = (pnl - _fc) - pos * ref  # NET of flip costs vs market/zero
        s = min(abs(ex) / 0.02, 1.0)
        # short positions: profit must reinforce SHORT (avoid side), so swap
        # valence - depress approach on short-profit, avoid on short-loss
        _sgn = -1.0 if pos < 0 else 1.0
        if ex * _sgn > 0:
            b.train(odor=_fe, reward=s)
        elif ex * _sgn < 0:
            b.train(odor=_fe, punish=s * punish_scale)
        prev_feat = feat
        hist.append((feat.copy(), float(pos), float(_fc)))
        # sleep consolidation after BIG days only (not calendar): today's
        # lesson is tag-protected, SHY washes old noise (uses sleep machinery)
        if sleep_big > 0 and abs(ex) > sleep_big:
            b.sleep(1)
        # hippocampal-style replay: buffer strong outcomes, replay top few
        # by |signal| every 20 days, then clear (oneirogenesis per cycle)
        if replay_top > 0 and abs(ex) > 0.01:
            _rbuf.append((feat.copy(),
                          s if ex * _sgn > 0 else 0.0,
                          s * punish_scale if ex * _sgn < 0 else 0.0))
        if replay_top > 0 and t % 20 == 0 and _rbuf:
            _rbuf.sort(key=lambda x: -(abs(x[1]) + abs(x[2])))
            for _ff, _rr, _pp in _rbuf[:replay_top]:
                _fm = _ff if _nomap else place(_ff, _live)
                if _rr > 0:
                    b.train(odor=_fm, reward=_rr)
                elif _pp > 0:
                    b.train(odor=_fm, punish=_pp)
            _rbuf.clear()
        # multi-day holding-period teaching (delayed conditioning, causal:
        # P&L of the position held since t-H attributed to feat_{t-H}).
        # excess framing vs always-long benchmark: (pos-1)*mkt_H rewards
        # dodging down markets, punishes sitting out rallies.
        if horizon > 1 and len(hist) > horizon:
            _hf, _hp, _ = hist[-horizon - 1]
            _mktH = cl[t] / cl[t - horizon] - 1
            _fcH = sum(x[2] for x in hist[-horizon - 1:])
            _hex = (_hp * _mktH - _fcH) if not excess else ((_hp - 1.0) * _mktH - _fcH)
            _ss = min(abs(_hex) / (0.02 * horizon), 1.0)
            if _hex > 0:
                b.train(odor=_hf if _nomap else place(_hf, _live), reward=_ss)
            elif _hex < 0:
                b.train(odor=_hf if _nomap else place(_hf, _live), punish=_ss * punish_scale)
        if sleep_every > 0 and (t - t0) % sleep_every == 0 and t > t0:
            b.sleep(2)
    rets_fly = np.array(rets_fly)
    bh = cl[_in[-1] + 1] / cl[_in[0]] if years else cl[-1] / cl[t0]
    tot = eq
    sh = float(rets_fly.mean() / (rets_fly.std() + 1e-12) * np.sqrt(252))
    run = np.maximum.accumulate(eqs)
    dd = float(((run - eqs) / run).max())
    hit = float((rets_fly > 0).mean())
    if verbose:
        print(f"{sym} {years or 'all'}: fly x{tot:.2f}  BH x{bh:.2f}  Sharpe {sh:+.2f}  "
              f"maxDD {dd:.1%}  hit {hit:.1%}  (n={len(rets_fly)})", flush=True)
    return {"tot": tot, "bh": bh, "sharpe": sh, "dd": dd, "hit": hit, "eq": eqs,
            "brain": b}


def run_ensemble(sym, configs, years=None, seed=1, verbose=True, wwin=60):
    """Colony with PERFORMANCE-weighted vote: each member hypothetical P&L
    (own vote x market) over trailing wwin days -> Sharpe -> softmax
    weights. Good members lead, bad ignored (equal votes dilute to null).
    Position in [0,1], |dpos| costs."""
    from fly_api import FlyBrainAPI as _FB
    cl = load(sym)
    rets = cl[1:] / cl[:-1] - 1
    yrs = load_ohlc(sym)["yr"]
    t0 = 25
    brains = []
    for cf in configs:
        b = _FB(mode="mb", path=".", seed=seed)
        if cf.get("leak", 0.0) > 0:
            b.set_state(cf["leak"])
        brains.append((b, cf, []))
    d = load_ohlc(sym)
    _in = [i for i in range(t0, len(cl) - 1)
           if years is None or int(yrs[i]) in years]
    pos = 0.0
    eq = 1.0
    eqs = []
    rets_fly = []
    pos_hist = []  # shared executed positions (causal attribution base)
    hyp = [[] for _ in brains]  # per-member hypothetical daily P&L
    for t in _in:
        votes = []
        feats = []
        for b, cf, hist in brains:
            feat = features2(d, t) if cf.get("ohlc") else features(cl, t)
            feats.append(feat)
            _lv = live_slots(b, need=len(feat))
            cf["_lv"] = _lv
            o = b.step(odor=place(feat, _lv))
            _pref = o["MB_app"] - o["MB_avo"]
            _thr = float(np.median(hist)) if len(hist) >= 20 else 0.0
            votes.append(1.0 if _pref > _thr else 0.0)
            hist.append(_pref)
            del hist[:-500]
        _sh = []
        for hh in hyp:
            w = hh[-wwin:]
            _sh.append(float(np.mean(w) / (np.std(w) + 1e-9)) if len(w) >= 20 else 0.0)
        _mx = max(_sh)
        _ew = [np.exp(min(max(s - _mx, -5.0), 5.0)) for s in _sh]
        _sw = sum(_ew)
        _w = [e / _sw for e in _ew]
        new_pos = float(sum(v * x for v, x in zip(votes, _w)))
        eq *= (1 - COST * abs(new_pos - pos))
        pos = new_pos
        pos_hist.append(pos)
        r = rets[t]
        pnl = pos * r
        eq *= (1 + pnl)
        eqs.append(eq)
        rets_fly.append(pnl)
        for hh, v in zip(hyp, votes):
            hh.append(v * r)
            if len(hh) > 90:
                del hh[:-90]
        # each brain: daily excess + own-horizon teaching on the SHARED
        # position outcome attributed to its OWN features/history
        for (b, cf, hist), feat in zip(brains, feats):
            ref = r
            ex = pnl - pos * ref
            s = min(abs(ex) / 0.02, 1.0)
            if ex > 0:
                b.train(odor=place(feat, cf["_lv"]), reward=s)
            elif ex < 0:
                b.train(odor=place(feat, cf["_lv"]), punish=s)
            H = int(cf.get("horizon", 1))
            fhist = cf.setdefault("_fh", [])
            fhist.append(place(feat, cf["_lv"]))
            del fhist[:-60]
            if H > 1 and len(fhist) > H and len(pos_hist) > H:
                _mktH = cl[t] / cl[t - H] - 1
                _hp = pos_hist[-H - 1]
                _hex = (_hp - 1.0) * _mktH  # vs always-long benchmark
                _ss = min(abs(_hex) / (0.02 * H), 1.0)
                if _hex > 0:
                    b.train(odor=fhist[-H - 1], reward=_ss)
                elif _hex < 0:
                    b.train(odor=fhist[-H - 1], punish=_ss)
    rets_fly = np.array(rets_fly)
    bh = cl[_in[-1] + 1] / cl[_in[0]]
    tot = eq
    sh = float(rets_fly.mean() / (rets_fly.std() + 1e-12) * np.sqrt(252))
    run = np.maximum.accumulate(eqs)
    dd = float(((run - eqs) / run).max())
    if verbose:
        print(f"{sym} {years or 'all'} ENSEMBLE[{len(brains)}]: fly x{tot:.2f}  BH x{bh:.2f}  "
              f"Sharpe {sh:+.2f}  maxDD {dd:.1%}  (n={len(rets_fly)})", flush=True)
    return {"tot": tot, "bh": bh, "sharpe": sh, "dd": dd, "eq": eqs}


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
    # champion: CORE-6 momentum features mapped onto KC-feeding ALPN slots
    # (dead slots waste features - only 5/12 first slots reach KC) + excess
    # + relative rule + 5-day horizon teaching + leak 0.3 memory.
    CH = dict(excess=True, rel_rule=True, horizon=6, leak=0.3, ohlc=False)
    for sym in (sys.argv[1:] or ["spy", "qqq"]):
        baselines(sym)
        run_market(sym, **CH)
