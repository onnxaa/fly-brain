"""Path following test: can the brain learn to steer along a trail?

Closed loop: render 64x64 view of a dark trail from the agent pose ->
full-brain step -> X_steer readout -> heading update -> move.
Usage: python3 test_path.py train    (teaches X_steer, saves w_path.npz)
       python3 test_path.py rollout  (loads weights, straight + sine path)
X_steer target: line offset u in [-1,1] (right +) -> steer u (turn right +).
"""
import sys
import numpy as np
from fly_api import FlyBrainAPI

WPATH = "w_path.npz"
IMG = 64


def straight_img(u, width=5, bg=0.0, fg=1.0):
    """Straight vertical BRIGHT trail at column 32*(1+u) on dark ground.

    Convention is deliberately bright-on-dark: the trail must be carried by
    ACTIVE cells. A dark line on gray ground encodes position by ABSENCE
    (shared bg drives all R identically) and no Hebbian/delta rule on active
    inputs can sculpt differential tuning from that (measured: flat +20)."""
    img = np.full((IMG, IMG), bg, np.float32)
    c = int(round(32 * (1 + u)))
    img[:, max(0, c - width // 2):c + width // 2 + 1] = fg
    return img


def render(x, y, th, path_fn, D=10.0, W=5.0, width=0.6, bg=0.0, fg=1.0):
    """Agent-frame view: rows = lookahead 0..D, cols = lateral -W..W."""
    rr = (np.arange(IMG) + 0.5) / IMG * D
    cc = (np.arange(IMG) - 31.5) / 32 * W
    dd, ll = np.meshgrid(rr, cc, indexing="ij")
    a = np.array([np.sin(th), np.cos(th)])
    n = np.array([np.cos(th), -np.sin(th)])
    xw = x + a[0] * dd + n[0] * ll
    yw = y + a[1] * dd + n[1] * ll
    online = np.abs(xw - path_fn(yw)) < width / 2
    return np.where(online, fg, bg).astype(np.float32)


def make_api():
    api = FlyBrainAPI(mode="full", path=".")
    api.enable_scaling()
    #Opponent steering (biology's solution for signed quantities with
    #non-negative firing rates): X_steerL learns max(0,-u), X_steerR max(0,u);
    #steer = R-L. Single signed output cannot work: the readout is a
    #non-negative sum, so negative targets are unreachable and teaching only
    #floors weights (measured: flat +20 / U-shaped +3).
    #R src (raw line position) + wscale=0.01: init responses must start at
    #target scale. Measured: wscale=0.05 inits at +2.5 while targets live at
    #0..0.75 - then every err is negative, everything sinks together (~8%/
    #trial) and differential sculpting never gets a word in (offline-proven:
    #uniform sink, init shape preserved). At matched scale the same rule
    #converges in a few rounds (cf. mb value tests).
    api.x_add_output("steerL", n=8, seed=21, src_pool=api.R, wscale=0.01)
    api.x_add_output("steerR", n=8, seed=22, src_pool=api.R, wscale=0.01)
    return api


def steer_of(api, img):
    o = api.step(image=img)
    return o["X_steerR"] - o["X_steerL"], o["X_steerR"], o["X_steerL"]


def train():
    api = make_api()
    offs = (-0.75, -0.375, 0.0, 0.375, 0.75)
    for t in range(5):
        for u in offs:
            img = straight_img(u)
            api.x_teach_output("steerL", target=max(0.0, -u), trials=1,
                               eta=1.2, image=img)
            api.x_teach_output("steerR", target=max(0.0, u), trials=1,
                               eta=1.2, image=img)
        print(f" round {t + 1}/5", flush=True)
    for u in (-0.5625, -0.375, 0.0, 0.375, 0.5625):
        s, r, l = steer_of(api, straight_img(u))
        print(f"u={u:+.3f} steer={s:+.3f} (R={r:.3f} L={l:.3f})", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def rollout(steer_on=True, path="sine", steps=70, k=0.6, v=0.6):
    api = make_api()
    api.load_weights(WPATH)
    if path == "sine":
        fn = lambda y: 3.0 * np.sin(2 * np.pi * y / 60.0)
    else:
        fn = lambda y: np.zeros_like(y) if isinstance(y, np.ndarray) else 0.0
    x, y, th = -2.0, 0.0, 0.0
    errs = []
    for _ in range(steps):
        img = render(x, y, th, fn)
        s = steer_of(api, img)[0] if steer_on else 0.0
        th += k * s
        x += v * np.sin(th)
        y += v * np.cos(th)
        fy = float(fn(y))
        errs.append(abs(x - fy))
    errs = np.array(errs)
    tag = "STEER" if steer_on else "BASE"
    print(f"{tag} {path}: RMS={errs.mean():.2f} max={errs.max():.2f} "
          f"off4={(errs > 4).mean() * 100:.0f}%", flush=True)
    return errs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    else:
        for p in ("straight", "sine"):
            rollout(True, p)
            rollout(False, p)


if __name__ == "__main__":
    main()
