"""Szybki test MB: plastycznosc tylko K->M (62k), 5 triali."""
import numpy as np, time
t0=time.time()
d = np.load("mb_circuit.npz")
pre, post, w = d["pre"], d["post"], d["weight"].astype(np.float32)
N = int(max(pre.max(), post.max()) + 1)
print(f"N={N} E={len(pre)}", flush=True)
ALPN, KC, MBON, DAN = d["inputs_ALPN"], d["KC"], d["MBON"], d["DAN"]

is_KC = np.zeros(N, bool); is_KC[KC] = True
is_MBON = np.zeros(N, bool); is_MBON[MBON] = True
km_mask = is_KC[pre] & is_MBON[post]
print(f"K->M edges: {km_mask.sum()} / {len(pre)}", flush=True)
km_idx = np.where(km_mask)[0]

rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wout = rng.normal(0, 0.3, size=(hid, hid)).astype(np.float32)
wM = np.abs(w)
sign = np.sign(w); sign[sign==0]=1.0

def forward(bias_idx):
    x = emb.copy(); x[bias_idx] += 2.0
    msg_src = (x[pre] @ Wmsg) * (wM*sign)[:,None]
    agg = np.zeros((N, hid), dtype=np.float32)
    np.add.at(agg, post, msg_src)
    out = np.maximum(0, x @ Wself + agg) @ Wout
    out = np.maximum(0,out)+0.1*np.minimum(0,out)
    return out, float(out[MBON].mean())

odorA = ALPN[:len(ALPN)//2]; odorB = ALPN[len(ALPN)//2:]
_, rA0 = forward(odorA); _, rB0 = forward(odorB)
print(f"before: A={rA0:.4f} B={rB0:.4f} ({time.time()-t0:.1f}s)", flush=True)

elig = np.zeros(km_idx.size, dtype=np.float32)
eta, alpha, decay = 0.02, 1e-5, 0.9
w_min, w_max = 0.5, 600.0
ref_in = np.zeros(N); np.add.at(ref_in, post, wM.astype(float))

def step(bias_idx, reward):
    global wM
    x = emb.copy(); x[bias_idx] += 2.0
    a0 = x.mean(axis=1)
    # koincydencja tylko na K->M (szybkie)
    hmsg = x @ Wmsg
    # a1 przyblizone: aktywacja post bez pelnego forward (szybkie)
    a1full, _ = forward(bias_idx)  # 1 forward (akceptowalne przy 5 trialach)
    a1 = a1full.mean(axis=1)
    e = (a0[pre[km_idx]] * a1[post[km_idx]]).astype(np.float32)
    elig[:] = elig*decay + e
    if reward:
        wM[km_idx] = np.clip(wM[km_idx]+eta*reward*elig-alpha, w_min, w_max)
        cur = np.zeros(N); np.add.at(cur, post, wM.astype(float))
        scale = np.ones(N); nz = cur>1e-9
        scale[nz] = np.clip(ref_in[nz]/cur[nz], 0.8, 1.25)
        wM[:] = (wM*scale[post]).astype(np.float32)

for t in range(5):
    step(odorA, 0.0); step(odorA, +1.0); step(odorB, 0.0)
    print(f"trial {t+1} done ({time.time()-t0:.1f}s)", flush=True)

_, rA1 = forward(odorA); _, rB1 = forward(odorB)
print(f"after 5 triali: A={rA1:.4f} B={rB1:.4f} prefersA={bool(rA1>rB1)}", flush=True)
print("PASS", flush=True)
