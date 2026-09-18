"""Fly as a language model (honest animal-cognition paradigms, no tokenizer).

MEASURED (mb, deterministic):
- Reber grammar, LENGTH split (train<=7, test>=8): fly test 90.4% (val 85.2%)
  vs bigram 88.0% vs chance 50%. Sparse placed codes were the fix (dense
  0.25 codes collided: 72.3%). Leak value inert for decisions (magnitudes
  move, signs don't - saturated valence margins).
- Next-symbol (X-zone transition model): 60.7% (strength scoring; argmax
  52.8%). Weak - thin output margins.
- Paired associates: n=4: 50%, n=12: 33%, n=20: 20% (chance 20%).
  Interference-limited; more negatives HURT (15%: global silencing).
  Binary X capacity is small-n only (cf. test-x 2-output PASS).

1. REBER-style artificial grammar: finite-state strings over {A,B,C,D};
   fly sees symbols as fixed sparse odor codes THROUGH TIME (leak carry),
   gets reward (grammatical) / punish (random strings, matched length).
   Tested on DISJOINT novel strings (generalization, not memorization).
   Baselines: chance 50%, bigram counter.
2. PAIRED ASSOCIATES: cue->recall capacity (MB heteroassociation core).
3. NEXT-SYMBOL: predict next symbol via per-symbol X readouts (if 1-2 pass).

Modes: mb (fast). Deterministic (seeded codes + brain).
"""
import numpy as np
from fly_api import FlyBrainAPI

SIGMA = "ABCDE"
# finite-state grammar (documented; paradigm matters, not the graph):
#   S -A->A, S -B->B | A -A->A, A -B->C, A -C->D | B -A->C, B -B->B, B -D->D |
#   C -A->E, C -C->C, C -E->END | D -B->E, D -D->D, D -E->END |
#   E -A->A, E -C->C, E -E->END ; lengths 3..12, hundreds of strings.
# Generalization test = LENGTH split (train<=7, test>=8): systematicity.
TRANS = {
    ("S", "A"): "A", ("S", "B"): "B",
    ("A", "A"): "A", ("A", "B"): "C", ("A", "C"): "D",
    ("B", "A"): "C", ("B", "B"): "B", ("B", "D"): "D",
    ("C", "A"): "E", ("C", "C"): "C", ("C", "E"): "END",
    ("D", "B"): "E", ("D", "D"): "D", ("D", "E"): "END",
    ("E", "A"): "A", ("E", "C"): "C", ("E", "E"): "END",
}


def gen_grammatical(rng, n):
    out = []
    while len(out) < n:
        st, s = "S", ""
        while len(s) < 8:
            opts = [(a, q) for (p, a), q in TRANS.items() if p == st]
            if not opts:
                break
            a, st = opts[rng.integers(len(opts))]
            s += a
            if st == "END" and len(s) >= 3:
                out.append(s)
                break
    return out


def gen_random(rng, n, lens):
    alpha = list(SIGMA)
    return ["".join(alpha[rng.integers(4)] for _ in range(l))
            for l in lens]


def is_grammatical(s):
    st = "S"
    for ch in s:
        st = TRANS.get((st, ch), None)
        if st is None:
            return False
    return st == "END"


def sym_codes(seed=11, dim=64, dens=0.10):
    # SPARSE codes (dense 0.25 collides: KC overlap kills separation) -
    # placed onto KC-feeding ALPN slots (dead-slot lesson) via place().
    rng = np.random.default_rng(seed)
    return {ch: (rng.random(dim) < dens).astype(np.float32)
            for ch in SIGMA}


def live_slots_mb(b, need=64):
    isK = np.zeros(b.N, bool)
    isK[b.KC] = True
    live = [i for i in range(len(b.ALPN))
            if b.wM[(b.pre == int(b.ALPN[i])) & isK[b.post]].sum() > 0]
    assert len(live) >= need, len(live)
    return live


def place(v, live):
    m = np.zeros(max(live[:len(v)]) + 1, dtype=np.float32)
    for j, f in enumerate(v):
        m[live[j]] = f
    return m


def _eval(b, codes, strings, want_pos):
    hits = 0
    for s in strings:
        b.reset_state()
        for ch in s:
            o = b.step(odor=codes[ch])
        hits += (o["MB_app"] > o["MB_avo"]) == want_pos
    return hits


def run_reber(ntrain=150, ntest=60, leak=0.5, seed=1, verbose=True,
              train_maxlen=7, test_minlen=8):
    rng = np.random.default_rng(seed)
    codes = sym_codes()
    b = FlyBrainAPI(mode="mb", path=".", seed=seed)
    b.set_state(leak)
    _live = live_slots_mb(b)
    codes = {ch: place(v, _live) for ch, v in codes.items()}
    # LENGTH split: train short (<=7), test long (>=8) = systematicity
    train_pos = [s for s in gen_grammatical(rng, ntrain * 6)
                 if len(s) <= train_maxlen][:ntrain]
    train_neg = [s for s in gen_random(rng, ntrain * 6,
                                       [3 + (i % 5) for i in range(ntrain * 6)])
                 if not is_grammatical(s)][:ntrain]
    # resolved negatives must be truly ungrammatical
    train_neg = [s if not is_grammatical(s) else s[:-1] + ("A" if s[-1] != "A" else "B")
                 for s in train_neg]
    for s in train_pos:
        b.reset_state()
        for ch in s:
            b.step(odor=codes[ch])
        b.train(odor=codes[s[-1]], reward=1.0)
    for s in train_neg:
        b.reset_state()
        for ch in s:
            b.step(odor=codes[ch])
        b.train(odor=codes[s[-1]], punish=1.0)
    # VALIDATION + TEST from LONG strings only (>= test_minlen), disjoint
    def longs(n, pool):
        return [s for s in gen_grammatical(rng, n * 6)
                if len(s) >= test_minlen and s not in pool][:n]
    val_pos = longs(ntest, set(train_pos))
    _vp = set(train_pos) | set(val_pos)
    test_pos = longs(ntest, _vp)
    _allp = _vp | set(test_pos)
    def negs(n, pool, lo):
        return [s for s in gen_random(rng, n * 8,
                                      [lo + (i % 5) for i in range(n * 8)])
                if not is_grammatical(s) and s not in pool][:n]
    val_neg = negs(ntest, set(train_neg), test_minlen)
    test_neg = negs(ntest, set(train_neg) | set(val_neg), test_minlen)
    vacc = (_eval(b, codes, val_pos, True) + _eval(b, codes, val_neg, False)) / (len(val_pos) + len(val_neg))
    acc = (_eval(b, codes, test_pos, True) + _eval(b, codes, test_neg, False)) / (len(test_pos) + len(test_neg))
    # bigram baseline, Laplace-smoothed log-prob, midpoint threshold
    from collections import Counter
    bi = Counter(); uni = Counter()
    for s in train_pos:
        for ch in s:
            uni[ch] += 1
        for a, c in zip(s, s[1:]):
            bi[(a, c)] += 1
    V = len(SIGMA)

    def lscore(s):
        sc = 0.0
        for a, c in zip(s, s[1:]):
            sc += np.log((bi[(a, c)] + 1) / (uni[a] + V))
        return sc / max(len(s) - 1, 1)

    _mp = np.mean([lscore(s) for s in train_pos])
    _mn = np.mean([lscore(s) for s in train_neg])
    _th = (_mp + _mn) / 2
    bh = (sum(lscore(s) > _th for s in test_pos) +
          sum(lscore(s) <= _th for s in test_neg))
    bacc = bh / (len(test_pos) + len(test_neg))
    if verbose:
        print(f"reber leak={leak}: fly val {vacc:.1%} test {acc:.1%}  bigram {bacc:.1%}  chance 50%  "
              f"(train {len(train_pos)}+{len(train_neg)})",
              flush=True)
    return acc, bacc


def run_next(ntrain=150, ntest=60, leak=0.5, seed=1, verbose=True,
             train_maxlen=7, test_minlen=8):
    """Next-symbol transition model in X-zone: 5 outputs (one per symbol),
    taught on (prefix-carry -> next) pairs from SHORT strings; scored on
    LONG strings (length generalization). Score = mean readout strength of
    the actual next symbol; threshold = midpoint of train-class means."""
    import numpy as _np
    rng = np.random.default_rng(seed)
    codes = sym_codes()
    b = FlyBrainAPI(mode="mb", path=".", seed=seed)
    b.set_state(leak)
    _live = live_slots_mb(b)
    codes = {ch: place(v, _live) for ch, v in codes.items()}
    for ch in SIGMA:
        b.x_add_output("nx" + ch, n=16, seed=seed)
    train_pos = [s for s in gen_grammatical(rng, ntrain * 6)
                 if len(s) <= train_maxlen][:ntrain]
    train_neg = [s for s in gen_random(rng, ntrain * 6,
                                       [3 + (i % 5) for i in range(ntrain * 6)])
                 if not is_grammatical(s)][:ntrain]
    for s in train_pos + train_neg:
        for i in range(len(s) - 1):
            pre, nxt = s[:i + 1], s[i + 1]
            b.reset_state()
            for ch in pre:
                b.step(odor=codes[ch])
            b.x_teach_output("nx" + nxt, trials=1, eta=0.5, high=True,
                             odor=codes[nxt])
            _w = [c for c in SIGMA if c != nxt][rng.integers(4)]
            b.reset_state()
            for ch in pre:
                b.step(odor=codes[ch])
            b.x_teach_output("nx" + nxt, trials=1, eta=0.5, high=False,
                             odor=codes[_w])

    def score(st):
        sc = []
        for i in range(len(st) - 1):
            pre, nxt = st[:i + 1], st[i + 1]
            b.reset_state()
            for ch in pre:
                b.step(odor=codes[ch])
            o = b.step(odor=codes[nxt])
            sc.append(o.get("X_nx" + nxt, 0.0))
        return float(_np.mean(sc)) if sc else 0.0

    _mp = _np.mean([score(s) for s in train_pos])
    _mn = _np.mean([score(s) for s in train_neg])
    _th = (_mp + _mn) / 2
    test_pos = [s for s in gen_grammatical(rng, ntest * 6)
                if len(s) >= test_minlen and s not in train_pos][:ntest]
    test_neg = [s for s in gen_random(rng, ntest * 6,
                                      [test_minlen + (i % 5) for i in range(ntest * 6)])
                if not is_grammatical(s)][:ntest]
    hits = (sum(score(s) > _th for s in test_pos) +
            sum(score(s) <= _th for s in test_neg))
    acc = hits / (len(test_pos) + len(test_neg))
    if verbose:
        print(f"next leak={leak}: fly {acc:.1%}  (train {len(train_pos)}+{len(train_neg)}, "
              f"test {len(test_pos)}+{len(test_neg)})", flush=True)
    return acc


def run_pairs(n=20, seed=1, verbose=True):
    """Paired associates (MB home turf): cue->target heteroassociation via
    X outputs; recall = argmax output given cue. Chance 20%."""
    rng = np.random.default_rng(seed)
    codes = sym_codes()
    b = FlyBrainAPI(mode="mb", path=".", seed=seed)
    b.set_state(0.0)
    _live = live_slots_mb(b)
    codes = {ch: place(v, _live) for ch, v in codes.items()}
    for ch in SIGMA:
        b.x_add_output("nx" + ch, n=16, seed=seed)
    pairs = [(c, t) for c in SIGMA for t in SIGMA if c != t]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:n]]
    for c, t in pairs:
        b.reset_state()
        b.x_teach_output("nx" + t, trials=2, eta=0.5, high=True, odor=codes[c])
        for w in [x for x in SIGMA if x != t][:2]:
            b.reset_state()
            b.x_teach_output("nx" + t, trials=1, eta=0.5, high=False, odor=codes[w])
    hits = 0
    for c, t in pairs:
        b.reset_state()
        o = b.step(odor=codes[c])
        pred = max(SIGMA, key=lambda k: o.get("X_nx" + k, 0.0))
        hits += pred == t
    acc = hits / len(pairs)
    if verbose:
        print(f"pairs n={n}: fly {acc:.1%}  chance 20%", flush=True)
    return acc


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "next":
        run_next()
    elif len(sys.argv) > 1 and sys.argv[1] == "pairs":
        run_pairs()
    else:
        run_reber()
