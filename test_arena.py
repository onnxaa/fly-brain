"""2D arena: food (odor + bright dot) vs predator (bright looming disk).

Brain: X_eat (odor conc -> 0..1, KC+R src), X_fleeL/R (opponent loom
bearing/size, R+ME+LO src), X_foodL/R (opponent dot bearing, R src).
Arbiter (spinal, outside brain): flee if max(flee)>0.15 (turn away,
speed x2), else feed (visual opponent + odor tropotaxis toward food).
Eat -> api.train(reward) online. Caught (d<1.0) ends the episode.

Usage: train_eat | train_flee | train_fooddir | train_fooddir2 |
       train_flee2 | eval | rollout <episodes>
"""
import sys
import numpy as np
from fly_api import FlyBrainAPI

WPATH = "w_arena.npz"
FOOD_ODOR = "ethyl_hexanoate"
IMG = 64


def make_api():
    api = FlyBrainAPI(mode="full", path=".")
    api.enable_scaling()
    api.x_add_output("eat", n=8, seed=31,
                     src_pool=np.concatenate([api.KC, api.R]))
    api.x_add_output("fleeL", n=8, seed=32,
                     src_pool=np.concatenate([api.R, api.ME, api.LO]),
                     wscale=0.01)
    api.x_add_output("fleeR", n=8, seed=33,
                     src_pool=np.concatenate([api.R, api.ME, api.LO]),
                     wscale=0.01)
    # food bearing opponent pair (turn_olf is real but ~0.001 at naturalistic
    # faint concentrations - 100x too small to steer on; dedicated visual
    # readout in the proven bright-dot/opponent/scale-matched regime).
    api.x_add_output("foodL", n=8, seed=34, src_pool=api.R, wscale=0.01)
    api.x_add_output("foodR", n=8, seed=35, src_pool=api.R, wscale=0.01)
    return api


def food_vecs(api):
    """Unit-concentration lateral ORN vectors for FOOD_ODOR (positional)."""
    dd = np.load(f"{api.path}/door_odors.npz")
    oi, ov = dd[FOOD_ODOR + "_idx"], dd[FOOD_ODOR + "_val"].astype(np.float32)
    m = {int(i): float(v) for i, v in zip(oi, ov)}
    vL = np.array([m.get(int(g), 0.0) for g in api.ORN_L], np.float32)
    vR = np.array([m.get(int(g), 0.0) for g in api.ORN_R], np.float32)
    return vL, vR


def disk_img(cx, r, bg=0.0, fg=1.0):
    """Bright disk centered (32+cx, 32), radius r px. cx in [-32,32]."""
    img = np.full((IMG, IMG), bg, np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img[(yy - 32) ** 2 + (xx - 32 - cx) ** 2 <= r * r] = fg
    return img


def render_arena(x, y, th, foods, pred, D=12.0, W=6.0):
    """Agent-frame 64x64: rows lookahead 0..D, cols lateral -W..W."""
    rr = (np.arange(IMG) + 0.5) / IMG * D
    cc = (np.arange(IMG) - 31.5) / 32 * W
    dd, ll = np.meshgrid(rr, cc, indexing="ij")
    a = np.array([np.sin(th), np.cos(th)])
    n = np.array([np.cos(th), -np.sin(th)])
    xw = x + a[0] * dd + n[0] * ll
    yw = y + a[1] * dd + n[1] * ll
    img = np.zeros((IMG, IMG), np.float32)
    for (fx, fy) in foods:
        d = np.hypot(xw - fx, yw - fy)
        img[d < 0.5] = 1.0
    if pred is not None:
        px, py = pred
        d = np.hypot(xw - px, yw - py)
        prad = max(0.6, 14.0 / max(1.0, np.hypot(x - px, y - py)))
        img[d < prad] = 1.0
    return img


def odor_lr(api, vL0, vR0, x, y, th, foods):
    """Lateral concentrations from two antennae (0.5 lateral offset)."""
    n = np.array([np.cos(th), -np.sin(th)])
    out = []
    for s in (-0.5, 0.5):
        sx, sy = x + n[0] * s, y + n[1] * s
        c = 0.0
        for (fx, fy) in foods:
            c += 1.0 / (1.0 + ((sx - fx) ** 2 + (sy - fy) ** 2) / 36.0)
        out.append(min(1.0, c))
    return out


def train_eat():
    api = make_api()
    vL0, vR0 = food_vecs(api)
    concs = (0.05, 0.15, 0.3, 0.5, 0.75, 1.0)
    for t in range(4):
        for c in concs:
            api.x_teach_output("eat", target=c, trials=1, eta=1.2,
                               odor_left=vL0 * c, odor_right=vR0 * c,
                               image=disk_img(0, 2))
        print(f" round {t + 1}/4", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def train_flee():
    api = make_api()
    api.load_weights(WPATH)
    sizes = (6, 14, 28)
    for t in range(2):
        for r in sizes:
            api.x_teach_output("fleeL", target=r / 28.0, trials=1, eta=1.2,
                               image=disk_img(-16, r))
            api.x_teach_output("fleeR", target=r / 28.0, trials=1, eta=1.2,
                               image=disk_img(+16, r))
            api.x_teach_output("fleeL", target=r / 28.0, trials=1, eta=1.2,
                               image=disk_img(0, r))
            api.x_teach_output("fleeR", target=r / 28.0, trials=1, eta=1.2,
                               image=disk_img(0, r))
        print(f" round {t + 1}/2", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def train_fooddir():
    api = make_api()
    api.load_weights(WPATH)
    for t in range(3):
        api.x_teach_output("foodL", target=0.75, trials=1, eta=1.2,
                           image=disk_img(-16, 3))
        api.x_teach_output("foodR", target=0.0, trials=1, eta=1.2,
                           image=disk_img(-16, 3))
        api.x_teach_output("foodL", target=0.0, trials=1, eta=1.2,
                           image=disk_img(+16, 3))
        api.x_teach_output("foodR", target=0.75, trials=1, eta=1.2,
                           image=disk_img(+16, 3))
        api.x_teach_output("foodL", target=0.3, trials=1, eta=1.2,
                           image=disk_img(0, 3))
        api.x_teach_output("foodR", target=0.3, trials=1, eta=1.2,
                           image=disk_img(0, 3))
        print(f" round {t + 1}/3", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def train_fooddir2():
    """In-distribution opponent food steering: single rendered dot at
    (fx, 8) - the exact stimulus family rollout sees. Graded targets
    foodL/R = max(0,-/+u)*0.75 (u=fx/6), center 0.3/0.3. Order alternates
    per round to avoid last-write bias."""
    api = make_api()
    api.load_weights(WPATH)
    FX = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
    for t in range(3):
        order = FX if t % 2 == 0 else FX[::-1]
        for fx in order:
            img = render_arena(0.0, 0.0, 0.0, [(fx, 8.0)], None)
            u = fx / 6.0
            tL = max(0.0, -u) * 0.75 + (0.3 if fx == 0.0 else 0.0)
            tR = max(0.0, u) * 0.75 + (0.3 if fx == 0.0 else 0.0)
            if t % 2 == 0:
                api.x_teach_output("foodL", target=tL, trials=1, eta=1.2,
                                   image=img)
                api.x_teach_output("foodR", target=tR, trials=1, eta=1.2,
                                   image=img)
            else:
                api.x_teach_output("foodR", target=tR, trials=1, eta=1.2,
                                   image=img)
                api.x_teach_output("foodL", target=tL, trials=1, eta=1.2,
                                   image=img)
        print(f" round {t + 1}/3", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def train_flee2():
    """Opponent flee rebalance: escape drive max(fL,fR)~m at any bearing,
    lateralized away-turn (fL-fR). tL=m*(0.5-u/2), tR=m*(0.5+u/2),
    u=cx/16, center both m/2. Alternating order per round."""
    api = make_api()
    api.load_weights(WPATH)
    sizes = (8, 20)
    CX = (-16, -8, 0, 8, 16)
    for t in range(2):
        seq = list(sizes) if t % 2 == 0 else list(sizes)[::-1]
        cxs = CX if t % 2 == 0 else CX[::-1]
        for r in seq:
            m = r / 28.0
            for cx in cxs:
                u = cx / 16.0
                tL = m * (0.5 - 0.5 * u)
                tR = m * (0.5 + 0.5 * u)
                if t % 2 == 0:
                    api.x_teach_output("fleeL", target=tL, trials=1, eta=1.2,
                                       image=disk_img(cx, r))
                    api.x_teach_output("fleeR", target=tR, trials=1, eta=1.2,
                                       image=disk_img(cx, r))
                else:
                    api.x_teach_output("fleeR", target=tR, trials=1, eta=1.2,
                                       image=disk_img(cx, r))
                    api.x_teach_output("fleeL", target=tL, trials=1, eta=1.2,
                                       image=disk_img(cx, r))
        print(f" round {t + 1}/2", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def train_flee3():
    """Opponent flee on true-distribution rendered predator: predator at
    (px, dp), px in (-5,0,5), dp in (3,5,8). tL/tR = m*(0.5-/+u/2),
    u=px/5, m by distance (1.0/0.5/0.25). Alternating order per round."""
    api = make_api()
    api.load_weights(WPATH)
    PXS = (-5.0, 0.0, 5.0)
    DPS = ((3.0, 1.0), (5.0, 0.5), (8.0, 0.25))
    for t in range(2):
        pxs = PXS if t % 2 == 0 else PXS[::-1]
        dps = DPS if t % 2 == 0 else DPS[::-1]
        for dp, m in dps:
            for px in pxs:
                u = px / 5.0
                tL = m * (0.5 - 0.5 * u)
                tR = m * (0.5 + 0.5 * u)
                img = render_arena(0.0, 0.0, 0.0, [], (px, dp))
                if t % 2 == 0:
                    api.x_teach_output("fleeL", target=tL, trials=1, eta=1.2,
                                       image=img)
                    api.x_teach_output("fleeR", target=tR, trials=1, eta=1.2,
                                       image=img)
                else:
                    api.x_teach_output("fleeR", target=tR, trials=1, eta=1.2,
                                       image=img)
                    api.x_teach_output("fleeL", target=tL, trials=1, eta=1.2,
                                       image=img)
        print(f" round {t + 1}/2", flush=True)
    api.save_weights(WPATH)
    print("saved", WPATH, flush=True)


def evaluate():
    api = make_api()
    api.load_weights(WPATH)
    vL0, vR0 = food_vecs(api)
    for c in (0.1, 0.5, 1.0):
        o = api.step(odor_left=vL0 * c, odor_right=vR0 * c,
                     image=disk_img(0, 2))
        print(f"eat conc={c}: X_eat={o['X_eat']:.3f} turn_olf={o.get('turn_olf', 0):+.4f}",
              flush=True)
    o = api.step(odor_left=vL0, odor_right=vR0 * 0.2, image=disk_img(0, 2))
    print(f"food-left: turn_olf={o.get('turn_olf', 0):+.4f} (expect >0)", flush=True)
    for side in (-16, 0, +16):
        for r in (4, 28):
            o = api.step(image=disk_img(side, r))
            print(f"disk side={side:+d} r={r}: fleeL={o['X_fleeL']:.3f} "
                  f"fleeR={o['X_fleeR']:.3f}", flush=True)
    for side in (-16, 0, +16):
        o = api.step(image=disk_img(side, 3))
        print(f"dot side={side:+d}: foodL={o['X_foodL']:.3f} "
              f"foodR={o['X_foodR']:.3f}", flush=True)


def rollout(n_ep=3, steps=50):
    api = make_api()
    api.load_weights(WPATH)
    vL0, vR0 = food_vecs(api)
    rng = np.random.default_rng(11)
    for ep in range(n_ep):
        foods = [(float(rng.uniform(-15, 15)), float(rng.uniform(5, 18)))
                 for _ in range(3)]
        px, py = 12.0, 14.0
        x, y, th = 0.0, -14.0, 0.0
        eaten, caught = 0, False
        mind = 1e9
        for _ in range(steps):
            cL, cR = odor_lr(api, vL0, vR0, x, y, th, foods)
            img = render_arena(x, y, th, foods, (px, py))
            o = api.step(odor_left=vL0 * cL, odor_right=vR0 * cR, image=img)
            fl, fr = o["X_fleeL"], o["X_fleeR"]
            if max(fl, fr) > 0.15:
                th += float(np.clip(0.5 * (fl - fr), -0.5, 0.5))
                sp = 1.5
            else:
                dth = (2.0 * (o["X_foodR"] - o["X_foodL"])
                       + 1.5 * (cR - cL))
                th += float(np.clip(dth, -0.5, 0.5))
                sp = 0.7
            x += sp * np.sin(th)
            y += sp * np.cos(th)
            x = float(np.clip(x, -20, 20))
            y = float(np.clip(y, -20, 20))
            dp = float(np.hypot(x - px, y - py))
            mind = min(mind, dp)
            if dp < 1.0:
                caught = True
                break
            for f in list(foods):
                if float(np.hypot(x - f[0], y - f[1])) < 1.5:
                    foods.remove(f)
                    eaten += 1
                    api.train(odor=FOOD_ODOR, reward=1.0)
            ang = float(np.arctan2(x - px, y - py))
            px += 0.85 * np.sin(ang)
            py += 0.85 * np.cos(ang)
            if not foods:
                break
        print(f"ep{ep}: steps={_ + 1} eaten={eaten}/3 caught={caught} "
              f"min_pred={mind:.1f}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if mode == "train_eat":
        train_eat()
    elif mode == "train_flee":
        train_flee()
    elif mode == "train_fooddir":
        train_fooddir()
    elif mode == "train_fooddir2":
        train_fooddir2()
    elif mode == "train_flee2":
        train_flee2()
    elif mode == "train_flee3":
        train_flee3()
    elif mode == "eval":
        evaluate()
    elif mode == "rollout":
        rollout(int(sys.argv[2]) if len(sys.argv) > 2 else 3)


if __name__ == "__main__":
    main()
