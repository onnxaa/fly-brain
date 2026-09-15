"""Ethology battery: T-maze, Buridan arena, looming habituation, detour.

Modes (each standalone):
  tmaze      - Quinn-style differential olfactory conditioning in full
               mode with real DoOR odors: CS+ (methyl_salicylate) paired
               with punishment, CS- (ethyl_hexanoate) unpaired. Preference index
               PI = (safe-shock)/(|safe|+|shock|) from MB_pref, after 1 and
               3 pairings, then contingency reversal. Saves w_tmaze.npz.
  train_fix  - Opponent stripe-fixation readouts X_fixL/R with retinotopic
               pools (fixL draws R cells with R_cx<0.4, fixR R_cx>0.6 -
               random pools alias: sign flips between trained positions).
               Bright-on-dark full-field bars. Saves w_etho.npz.
  buridan    - Pacing Buridan on an LED-panorama render (world poles become
               bearing-column bars = the trained stimulus class): the lit
               pole alternates sides, gusts knock the heading, blind scan
               recovers a lost pole. Metric: facing% + re-face lags,
               STEER vs BASE.
  habituate  - 15x fixed looming disk on w_arena flee readouts: slope of
               escape over trials (stability control - rate-mode brain has
               no non-associative plasticity) + size-scaling control
               (response must still grow with disk size, i.e. flatness over
               trials is not saturation) + novel-bearing probe.
  detour     - Gotz-inspired re-acquisition: fixate target stripe (20
               steps), occlude it + show distractor (20 steps), restore
               target (20 steps). Metrics: capture by distractor (proves
               stimulus-driven steering, not ballistic) + return time.

Usage: python3 test_ethology.py tmaze|train_fix|buridan|habituate|detour
"""
import sys
import numpy as np
from fly_api import FlyBrainAPI

IMG = 64
W_ETHO = "w_etho.npz"
W_TMAZE = "w_tmaze.npz"
W_ARENA = "w_arena.npz"

CS_SHOCK = "methyl_salicylate"
CS_SAFE = "ethyl_hexanoate"


def pi_of(safe, shock):
    return float((safe - shock) / (abs(safe) + abs(shock) + 1e-9))


def tmaze():
    api = FlyBrainAPI(mode="full", path=".")
    api.enable_scaling()
    pre_s = api.step(odor=CS_SAFE)["MB_pref"]
    pre_p = api.step(odor=CS_SHOCK)["MB_pref"]
    print(f"naive: safe={pre_s:+.2f} shock={pre_p:+.2f} PI={pi_of(pre_s, pre_p):+.2f}",
          flush=True)
    api.train(odor=CS_SHOCK, punish=1.0)
    s1 = api.step(odor=CS_SAFE)["MB_pref"]
    p1 = api.step(odor=CS_SHOCK)["MB_pref"]
    print(f"1-trial: safe={s1:+.2f} shock={p1:+.2f} PI={pi_of(s1, p1):+.2f} (expect >0)",
          flush=True)
    for _ in range(4):
        api.train(odor=CS_SHOCK, punish=1.0)
    s3 = api.step(odor=CS_SAFE)["MB_pref"]
    p3 = api.step(odor=CS_SHOCK)["MB_pref"]
    print(f"5-trial: safe={s3:+.2f} shock={p3:+.2f} PI={pi_of(s3, p3):+.2f} (expect >0)",
          flush=True)
    for _ in range(3):
        api.train(odor=CS_SAFE, punish=1.0)
        api.train(odor=CS_SHOCK, reward=1.0)
    sr = api.step(odor=CS_SAFE)["MB_pref"]
    pr = api.step(odor=CS_SHOCK)["MB_pref"]
    print(f"reversed: safe={sr:+.2f} shock={pr:+.2f} PI={pi_of(sr, pr):+.2f} (expect <0)",
          flush=True)
    api.save_weights(W_TMAZE)
    print("saved", W_TMAZE, flush=True)


def bar_img(u, width=5, bg=0.0, fg=1.0):
    """Full-field bright vertical bar at column 32*(1+u)."""
    img = np.full((IMG, IMG), bg, np.float32)
    c = int(round(32 * (1 + u)))
    img[:, max(0, c - width // 2):c + width // 2 + 1] = fg
    return img


def render_pano(x, y, th, poles, width=5):
    """LED-panorama render: each world pole (px,py) ahead becomes a
    full-height bright vertical bar at its bearing column (binary
    on/off, like real LED arenas). This is exactly the stimulus class
    train_fix teaches (full-field bars) - zero train/test mismatch by
    construction. Poles behind (|u|>1) are invisible."""
    img = np.zeros((IMG, IMG), np.float32)
    ax, ay = float(np.sin(th)), float(np.cos(th))
    nx, ny = float(np.cos(th)), float(-np.sin(th))
    for (px, py) in poles:
        dx, dy = px - x, py - y
        fwd, lat = dx * ax + dy * ay, dx * nx + dy * ny
        if fwd <= 0.5:
            continue
        u = float(np.arctan2(lat, fwd)) / (np.pi / 2)
        if abs(u) > 1:
            continue
        c = int(round(32 * (1 + u)))
        img[:, max(0, c - width // 2):c + width // 2 + 1] = 1.0
    return img


def make_fix_api():
    # Retinotopic pools (data, not luck): fixL draws only R cells with
    # receptive fields in the left visual field (R_cx<0.4), fixR right
    # (R_cx>0.6). Left bars then drive fixL more from the start with the
    # correct sign; teaching only sculpts magnitudes. Random 8-neuron
    # pools have interleaved spatial tuning - the opponent difference
    # aliases (measured: sign flips between trained bar positions).
    api = FlyBrainAPI(mode="full", path=".")
    api.enable_scaling()
    R = np.asarray(api.R, dtype=np.int32)
    cx = np.asarray(api.R_cx, dtype=np.float64)
    api.x_add_output("fixL", n=8, seed=21,
                     src_pool=R[cx < 0.4], wscale=0.01)
    api.x_add_output("fixR", n=8, seed=22,
                     src_pool=R[cx > 0.6], wscale=0.01)
    return api


def train_fix():
    # Mirrors test_path's proven recipe exactly (5 rounds, L-then-R, plain
    # opponent targets): wide bars drive R broadly, overlapping codes need
    # the full dose - graded/alternating variants under-converge here.
    api = make_fix_api()
    o = api.step(image=bar_img(-0.5))
    print(f"naive u=-0.5: steer={o['X_fixR'] - o['X_fixL']:+.3f} "
          f"(expect <0 from retinotopy alone)", flush=True)
    UU = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)
    for t in range(3):
        for u in UU:
            img = bar_img(u)
            api.x_teach_output("fixL", target=max(0.0, -u), trials=1,
                               eta=1.2, image=img)
            api.x_teach_output("fixR", target=max(0.0, u), trials=1,
                               eta=1.2, image=img)
        print(f" round {t + 1}/3", flush=True)
    for u in (-0.5, -0.125, 0.0, 0.125, 0.5):
        o = api.step(image=bar_img(u))
        print(f"u={u:+.2f} fixL={o['X_fixL']:.3f} fixR={o['X_fixR']:.3f} "
              f"steer={o['X_fixR'] - o['X_fixL']:+.3f}", flush=True)
    api.save_weights(W_ETHO)
    print("saved", W_ETHO, flush=True)


def ang_dist(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(float(d))


def pole_bearing(x, y, th, pole):
    dx, dy = pole[0] - x, pole[1] - y
    ax, ay = float(np.sin(th)), float(np.cos(th))
    nx, ny = float(np.cos(th)), float(-np.sin(th))
    return ang_dist(th, float(np.arctan2(dx, dy)))


def buridan(steps=25, n_switch=3):
    # Pacing Buridan on the LED panorama: the lit pole alternates sides
    # every `steps` steps; the agent must re-fixate each time. Gusts
    # (uniform heading noise) rule out the trivial straight-line control
    # winning by wall-pinning. Blind scan (slow turn when no pole in the
    # ahead hemisphere) is harness-level search - flies saccade when they
    # lose the stripe; without it a >90 deg switch is unrecoverable for
    # ANY ahead-only visual system. Metrics: facing% + re-face lag after
    # each switch, STEER vs BASE.
    api = make_fix_api()
    api.load_weights(W_ETHO)
    rng = np.random.default_rng(7)
    for steer_on in (True, False):
        x, y, th = 0.0, -4.0, 0.0
        face, tot, lags = 0, 0, []
        lag = None
        for sw in range(n_switch):
            pole = (-4.0, 8.0) if sw % 2 == 0 else (4.0, 8.0)
            for i in range(steps):
                th += float(rng.uniform(-0.12, 0.12))
                img = render_pano(x, y, th, (pole,))
                if img.sum() == 0:
                    th += 0.15
                elif steer_on:
                    o = api.step(image=img)
                    s = o["X_fixR"] - o["X_fixL"]
                    s = 0.0 if abs(s) < 0.04 else s
                    th += float(np.clip(0.8 * s, -0.5, 0.5))
                x += 0.35 * np.sin(th)
                y += 0.35 * np.cos(th)
                x = float(np.clip(x, -9, 9))
                y = float(np.clip(y, -9, 9))
                e = pole_bearing(x, y, th, pole)
                tot += 1
                face += e < np.deg2rad(25)
                if lag is None:
                    if e < np.deg2rad(25):
                        lags.append(i + 1)
                        lag = i
                    elif i == steps - 1:
                        lags.append(None)
            lag = None
        tag = "STEER" if steer_on else "BASE"
        print(f"{tag}: facing={face / tot * 100:.0f}% "
              f"reface-lags={lags}", flush=True)


def disk_img(cx, r, bg=0.0, fg=1.0):
    img = np.full((IMG, IMG), bg, np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img[(yy - 32) ** 2 + (xx - 32 - cx) ** 2 <= r * r] = fg
    return img


def habituate(n=15):
    from test_arena import make_api
    api = make_api()
    api.load_weights(W_ARENA)
    seq = []
    for i in range(n):
        o = api.step(image=disk_img(0, 20))
        m = max(o["X_fleeL"], o["X_fleeR"])
        seq.append(m)
        if i in (0, n // 2, n - 1):
            print(f"trial {i + 1}: flee={m:.3f}", flush=True)
    seq = np.array(seq)
    slope = float(np.polyfit(np.arange(n), seq, 1)[0])
    print(f"slope={slope:+.4f}/trial (expect ~0: no non-associative "
          f"plasticity, escape stays reliable)", flush=True)
    for r in (4, 10, 20, 28):
        o = api.step(image=disk_img(0, r))
        print(f"size r={r}: flee={max(o['X_fleeL'], o['X_fleeR']):.3f}",
              flush=True)
    o = api.step(image=disk_img(16, 20))
    print(f"novel bearing: flee={max(o['X_fleeL'], o['X_fleeR']):.3f}",
          flush=True)


def detour(steps=20):
    api = make_fix_api()
    api.load_weights(W_ETHO)
    x, y, th = 0.0, -4.0, 0.0
    tgt = (4.0, 8.0)
    ret = None
    errs = {}
    for i in range(3 * steps):
        if i < steps:
            poles = (tgt,)
            phase = "target"
        elif i < 2 * steps:
            poles = ((-4.0, 8.0),)
            phase = "distractor"
        else:
            poles = (tgt,)
            phase = "return"
        img = render_pano(x, y, th, poles)
        if img.sum() == 0:
            th += 0.15
        else:
            o = api.step(image=img)
            s = o["X_fixR"] - o["X_fixL"]
            s = 0.0 if abs(s) < 0.04 else s
            th += float(np.clip(0.8 * s, -0.5, 0.5))
        x += 0.35 * np.sin(th)
        y += 0.35 * np.cos(th)
        x = float(np.clip(x, -9, 9))
        y = float(np.clip(y, -9, 9))
        e = pole_bearing(x, y, th, tgt)
        if i in (steps - 1, 2 * steps - 1, 3 * steps - 1):
            errs[phase] = e
            print(f"end-{phase}: heading-err={np.rad2deg(e):.0f}deg",
                  flush=True)
        if phase == "return" and ret is None and e < np.deg2rad(25):
            ret = i - 2 * steps + 1
    e_end = np.rad2deg(pole_bearing(x, y, th, tgt))
    cap = np.rad2deg(errs.get("distractor", np.pi)) > 45
    print(f"captured-by-distractor={'yes' if cap else 'no'} "
          f"return-time={ret}/{steps} final-err={e_end:.0f}deg", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "tmaze"
    if mode == "tmaze":
        tmaze()
    elif mode == "train_fix":
        train_fix()
    elif mode == "buridan":
        buridan()
    elif mode == "habituate":
        habituate()
    elif mode == "detour":
        detour()
    else:
        print("unknown mode", flush=True)


if __name__ == "__main__":
    main()
