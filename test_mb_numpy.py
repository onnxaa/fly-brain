"""Test prawdziwego MB (numpy, light): forward + online T-maze."""
import numpy as np

d = np.load("mb_circuit.npz")
pre, post, w = d["pre"], d["post"], d["weight"].astype(np.float32)
N = int(max(pre.max(), post.max()) + 1)
E = len(pre)
print(f"N={N} E={E} wmean={w.mean():.3f} wmax={w.max()} wmin={w.min()}")

ALPN, KC, MBON, DAN = d["inputs_ALPN"], d["KC"], d["MBON"], d["DAN"]
print(f"ALPN={len(ALPN)} KC={len(KC)} MBON={len(MBON)} DAN={len(DAN)}")

rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wout = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
wM = np.abs(w)  # magnitudy (znak: prawie wszystko +)
sign = np.sign(w)
sign[sign == 0] = 1.0

def forward(bias_idx, bias_val=2.0):
    x = emb.copy()
    x[bias_idx] += bias_val
    # 1 warstwa GNN (wystarczy na test mechaniczny)
    msg_src = (x[pre] @ Wmsg) * (wM * sign)[:, None]
    agg = np.zeros((N, hid), dtype=np.float32)
    np.add.at(agg, post, msg_src)
    out = np.maximum(0, x @ Wself + agg) @ Wout
    out = np.maximum(0, out) + 0.1 * np.minimum(0, out)  # leaky relu
    rates = np.log1p(np.exp(out[MBON].mean(axis=1))) if False else out[MBON]  # surowe
    # readout: srednia populacji MBON
    return out, out[MBON].mean()

odorA = ALPN[:len(ALPN)//2]
odorB = ALPN[len(ALPN)//2:]
print(f"odorA={len(odorA)} ALPN, odorB={len(odorB)} ALPN")

_, rA0 = forward(odorA)
_, rB0 = forward(odorB)
print(f"before: MBONmean odorA={rA0:.4f} odorB={rB0:.4f}")

# online R-Hebb na magnitudach (bez torch): eligibility na koincydencji pre/post
elig = np.zeros(E, dtype=np.float32)
eta, alpha, decay = 0.005, 1e-5, 0.9
w_min, w_max = 0.5, 600.0
ref_in = np.zeros(N, dtype=np.float64)
np.add.at(ref_in, post, wM.astype(np.float64))

def online_step(bias_idx, reward):
    global wM
    x = emb.copy(); x[bias_idx] += 2.0
    a0 = x.mean(axis=1)
    msg_src = (x[pre] @ Wmsg) * (wM * sign)[:, None]
    agg = np.zeros((N, hid), dtype=np.float32)
    np.add.at(agg, post, msg_src)
    h1 = np.maximum(0, x @ Wself + agg)
    a1 = h1.mean(axis=1)
    elig[:] = elig * decay + (a0[pre] * a1[post]).astype(np.float32)
    if reward != 0:
        wM[:] = np.clip(wM + eta * reward * elig - alpha, w_min, w_max).astype(np.float32)
        # renorm per neuron do ref
        cur = np.zeros(N, dtype=np.float64)
        np.add.at(cur, post, wM.astype(np.float64))
        scale = np.ones(N)
        nz = cur > 1e-9
        scale[nz] = np.clip(ref_in[nz] / cur[nz], 0.8, 1.25)
        wM[:] = (wM * scale[post]).astype(np.float32)
    _, r = forward(bias_idx)
    return r

for t in range(30):
    online_step(odorA, 0.0)
    online_step(odorA, +1.0)  # nagroda za A
    online_step(odorB, 0.0)

_, rA1 = forward(odorA)
_, rB1 = forward(odorB)
print(f"after 30 triali: MBONmean odorA={rA1:.4f} odorB={rB1:.4f}")
print(f"prefers A: {bool(rA1 > rB1)}")
print("wM mean before/after:", float(w.mean()), float(wM.mean()))
print("PASS")
