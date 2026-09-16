"""WEIRD probe (mb, fast): non-ecological hard tasks for a frozen fly brain.

T1 2-bit XOR on graded ALPN drives (linearly inseparable!)
T2 3-bit parity (8 patterns, harder)
T3 contextual XOR across modalities (odor x touch conjunctions)
T4 nonlinear regression: a^2 and sin(2*pi*a) from graded drive
T5 mirror invariance: even function f(s)==f(-s), unseen symmetric pairs
NOT tested: temporal sequences (architecture is stateless - documented limit).
"""
import numpy as np
from fly_api import FlyBrainAPI


def stim2(a, b):
    v = np.zeros(20, np.float32)
    v[0:10] = a
    v[10:20] = b
    return dict(alpn=v)


def stim3(a, b, c):
    v = np.zeros(30, np.float32)
    v[0:10] = a
    v[10:20] = b
    v[20:30] = c
    return dict(alpn=v)


def main():
    R = {}
    # ---- T1 XOR ----
    b = FlyBrainAPI(mode="mb", path=".")
    b.x_add_output("xor", n=32, seed=11, per_in=256)
    pats = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for _ in range(12):
        for a, bb, t in pats:
            b.x_teach_output("xor", trials=2, eta=0.5, high=bool(t), **stim2(a, bb))
    got = [b.step(**stim2(a, bb))["X_xor"] for a, bb, _ in pats]
    print(f"T1 XOR: {[f'{g:.2f}' for g in got]} (tgt 0,1,1,0)", flush=True)
    R["T1"] = got[0] < 0.5 and got[3] < 0.5 and got[1] > 0.5 and got[2] > 0.5
    # ---- T2 parity-3 ----
    b.x_add_output("par", n=64, seed=12, per_in=256)
    pats3 = [(a, bb, c, (a + bb + c) % 2) for a in (0, 1) for bb in (0, 1) for c in (0, 1)]
    for _ in range(15):
        for a, bb, c, t in pats3:
            b.x_teach_output("par", trials=2, eta=0.5, high=bool(t), **stim3(a, bb, c))
    got3 = [b.step(**stim3(a, bb, c))["X_par"] for a, bb, c, _ in pats3]
    ok3 = sum(1 for g, (_, _, _, t) in zip(got3, pats3) if (g > 0.5) == bool(t))
    print(f"T2 parity3: {ok3}/8 correct", flush=True)
    R["T2"] = ok3 >= 7
    # ---- T3 contextual XOR (odor x mech) ----
    b2 = FlyBrainAPI(mode="mb", path=".")
    b2.x_add_output("ctx", n=16, seed=13, per_in=64)
    # mb has no MECH pool: use alpn slices as pseudo-touch
    def stimx(o, m):
        v = np.zeros(40, np.float32)
        v[0:20] = 1.0 if o == "A" else 0.0
        v[0:10] = 0.0 if o == "A" else 1.0
        v[20:40] = float(m)
        return dict(alpn=v)
    ctx = [(("A", 1), 1), (("A", 0), 0), (("B", 1), 0), (("B", 0), 1)]
    for _ in range(12):
        for (o, m), t in ctx:
            b2.x_teach_output("ctx", trials=2, eta=0.5, high=bool(t), **stimx(o, m))
    gotc = [b2.step(**stimx(o, m))["X_ctx"] for (o, m), _ in ctx]
    print(f"T3 ctx-XOR: {[f'{g:.2f}' for g in gotc]} (tgt 1,0,0,1)", flush=True)
    R["T3"] = gotc[0] > 0.5 and gotc[3] > 0.5 and gotc[1] < 0.5 and gotc[2] < 0.5
    # ---- T4 nonlinear regression ----
    # uniform graded drive gives nested codes -> readout can only be monotonic
    # in drive (a^2 works); oscillation needs distinct codes (thermometer).
    # Readout is nonnegative by construction -> offset targets to [0,1].
    b3 = FlyBrainAPI(mode="mb", path=".")
    b3.x_add_output("sq", n=24, seed=14, per_in=256)
    b3.x_add_output("sn", n=48, seed=15, per_in=256)
    grid = np.linspace(0, 1, 11)
    for _ in range(10):
        for a in grid:
            v = np.zeros(20, np.float32)
            v[:] = a
            b3.x_teach_output("sq", target=float(a * a), trials=1, eta=1.0, alpn=v)

    def therm(a, n=20):
        v = np.zeros(n, np.float32)
        v[:max(1, int(round(a * n)))] = 1.0
        return dict(alpn=v)

    for _ in range(10):
        for a in grid:
            b3.x_teach_output("sn", target=float((np.sin(2 * np.pi * a) + 1) / 2), trials=1, eta=1.0, **therm(a))
    test = np.linspace(0.05, 0.95, 10)
    esq, esn, tsq, tsn = [], [], [], []
    for a in test:
        v = np.zeros(20, np.float32)
        v[:] = a
        esq.append(b3.step(alpn=v)["X_sq"])
        esn.append(b3.step(**therm(a))["X_sn"] * 2 - 1)
        tsq.append(a * a)
        tsn.append(np.sin(2 * np.pi * a))
    mae_sq = float(np.abs(np.array(esq) - tsq).mean())
    corr_sn = float(np.corrcoef(esn, tsn)[0, 1])
    print(f"T4 a^2 MAE={mae_sq:.3f} sin-therm corr={corr_sn:.3f} (oscillation: correlation only, no amplitude)", flush=True)
    R["T4"] = mae_sq < 0.11 and corr_sn > 0.5
    # ---- T5 mirror invariance f(s)==f(-s) ----
    b4 = FlyBrainAPI(mode="mb", path=".")
    b4.x_add_output("sym", n=24, seed=16, per_in=256)

    def stims(s):
        v = np.zeros(20, np.float32)
        v[0:10] = max(s, 0)
        v[10:20] = max(-s, 0)
        return dict(alpn=v)

    train_s = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for _ in range(6):
        for s in train_s:
            b4.x_teach_output("sym", target=float(abs(s)), trials=1, eta=1.0, **stims(s))
    errs = []
    for s in (-0.75, -0.375, 0.375, 0.75):
        g = b4.step(**stims(s))["X_sym"]
        gm = b4.step(**stims(-s))["X_sym"]
        errs.append(abs(g - abs(s)) + abs(gm - abs(s)))
        print(f"T5 s={s:+.3f}: f={g:.3f} f(mirror)={gm:.3f} tgt={abs(s):.3f}", flush=True)
    print(f"T5 mean err={float(np.mean(errs)):.3f}", flush=True)
    R["T5"] = float(np.mean(errs)) < 0.2
    print("----", flush=True)
    for k in ["T1", "T2", "T3", "T4", "T5"]:
        print(f"{k}: {'PASS' if R[k] else 'FAIL'}", flush=True)
    print("PASS-weird" if all(R.values()) else "FAIL-weird", flush=True)


if __name__ == "__main__":
    main()
