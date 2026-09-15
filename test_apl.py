"""APL: kc = relu(drive - g*mean(drive)) before top-k. Sweep g on the MB circuit (fast)."""
import numpy as np, time
d = np.load("mb_circuit.npz")
pre, post, w = d["pre"], d["post"], d["weight"].astype(np.float32)
N = int(max(pre.max(), post.max())+1)
ALPN, KC = d["inputs_ALPN"], d["KC"]
rng = np.random.default_rng(1)
in_dim, hid = 8, 16
emb = rng.normal(0, 0.5, size=(N, in_dim)).astype(np.float32)
Wmsg = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
Wself = rng.normal(0, 0.3, size=(in_dim, hid)).astype(np.float32)
wM = np.abs(w); sign = np.sign(w); sign[sign==0]=1.0
oA = ALPN[:len(ALPN)//2]; oB = ALPN[len(ALPN)//2:]
xA = emb.copy(); xA[oA] += 2.0
xB = emb.copy(); xB[oB] += 2.0
def drive(x):
    msg = (x[pre] @ Wmsg)*(wM*sign)[:,None]
    agg = np.zeros((N, hid), dtype=np.float32); np.add.at(agg, post, msg)
    return (np.maximum(0, x @ Wself + agg))[KC].mean(axis=1)
dA, dB = drive(xA), drive(xB)
for g in [0.0, 0.5, 0.9, 1.0, 1.2, 1.5]:
    sA = np.maximum(0, dA - g*dA.mean()); sB = np.maximum(0, dB - g*dB.mean())
    k = max(1, int(len(sA)*0.05))
    mA = sA >= np.sort(sA)[-k]; mB = sB >= np.sort(sB)[-k]
    ov = np.logical_and(mA, mB).sum()
    corr = float(np.corrcoef(sA, sB)[0,1])
    print(f"g={g}: overlap={ov}/{k} corr={corr:.2f} deadA={(sA[mA]==0).sum()} ({(mA).sum()} active)", flush=True)
print("PASS-apl-sweep", flush=True)
