"""Arithmetic test: can the brain learn a+b (and a*b) from graded drives?

Operand encoding: alpn vector, ALPN[0:10]=a, ALPN[10:20]=b, a/b in [0,1].
Readouts: X_sum -> a+b, X_prod -> a*b (mb mode, fast).
Train on a 16-pair subset of the 5x5 grid, test held-out interpolation +
out-of-range extrapolation. PASS: held-out sum corr>0.9 and MAE<0.25.
"""
import numpy as np
from fly_api import FlyBrainAPI

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NENC = 20


def stim(a, b):
    v = np.zeros(NENC, np.float32)
    v[0:10] = float(a)
    v[10:20] = float(b)
    return dict(alpn=v)


def main():
    api = FlyBrainAPI(mode="mb", path=".")
    api.x_add_output("sum", n=16, seed=3, per_in=256)
    api.x_add_output("prod", n=16, seed=4, per_in=256)
    all_pairs = [(a, b) for a in GRID for b in GRID]
    evens = [p for i, p in enumerate(all_pairs) if i % 2 == 0]
    odds = [p for i, p in enumerate(all_pairs) if i % 2 == 1]
    train = (evens + odds)[:16]
    held = [p for p in all_pairs if p not in train]
    extra = [(1.25, 0.5), (0.5, 1.25), (1.25, 1.25)]
    print(f"train={len(train)} held={len(held)} extra={len(extra)}", flush=True)
    for t in range(10):
        for (a, b) in train:
            api.x_teach_output("sum", target=a + b, trials=1, eta=1.0, **stim(a, b))
            api.x_teach_output("prod", target=a * b, trials=1, eta=1.0, **stim(a, b))
        if (t + 1) % 5 == 0:
            print(f" round {t + 1}/10", flush=True)

    def ev(pairs, fn):
        got = np.array([api.step(**stim(a, b))[fn] for (a, b) in pairs])
        return got

    for name, tgt in (("X_sum", lambda a, b: a + b), ("X_prod", lambda a, b: a * b)):
        for split, pairs in (("train", train), ("held", held), ("extra", extra)):
            t = np.array([tgt(a, b) for (a, b) in pairs])
            g = ev(pairs, name)
            mae = float(np.abs(g - t).mean())
            corr = float(np.corrcoef(g, t)[0, 1]) if len(pairs) > 2 else float("nan")
            print(f"{name} {split}: MAE={mae:.3f} corr={corr:.3f}", flush=True)
            if split == "held":
                for (a, b), gg in zip(pairs, g):
                    print(f"   a={a:.2f} b={b:.2f} tgt={tgt(a,b):.2f} got={gg:.3f}", flush=True)
    gs = ev(held, "X_sum")
    ts = np.array([a + b for (a, b) in held])
    ok = float(np.corrcoef(gs, ts)[0, 1]) > 0.9 and float(np.abs(gs - ts).mean()) < 0.25
    print("PASS-arithmetic" if ok else "FAIL-arithmetic", flush=True)


if __name__ == "__main__":
    main()
