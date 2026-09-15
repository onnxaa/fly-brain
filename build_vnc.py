"""Build real VNC circuit from MANC v1.2.1 (Takemura/Marin/Cheong et al., eLife 2024).

IN (downloaded once, public GCS bucket):
  manc_meta.feather   (23,650 neurons, types/NT/side)
  manc_simple.feather (5,305,354 neuron->neuron edges: pre,post,count)
OUT:
  vnc_circuit.npz: pre/post (local 0..M-1 int32), weight=|count| f32,
    sign f32 from presynaptic NT (ACh=+1, GABA=-1, glutamate=-1 central,
    unknown/unclear=+1; motor glutamate kept +1 edge sign irrelevant, sinks),
    pools: desc, motor_all, leg_L/R, wing_L/R, neck, sensory, ascending,
    side, cell_type per neuron, id2idx map info.
  vnc_bridge.npz: brain DESC (global FAFB idx) -> VNC desc (local idx) pairs
    matched by exact DN cell_type name (196 shared types, 500/1299 brain,
    499/1322 VNC). Bridge weight = protocol (X-zone, default 1.0).
RULE: VNC topology/Dale frozen from MANC; only the bridge is protocol.
"""
import numpy as np
import pandas as pd

NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0,
           "unknown": 1.0, "unclear": 1.0}

print("load meta...", flush=True)
m = pd.read_feather("manc_meta.feather")
ids = m["manc_121_id"].values.astype(np.int64)
M = len(ids)
id2i = {int(v): i for i, v in enumerate(ids)}
print(f"M={M}", flush=True)

nt = m["neurotransmitter_predicted"].fillna("unknown").values
sign_pre = np.array([NT_SIGN.get(str(x).lower(), 1.0) for x in nt], dtype=np.float32)

cc = m["cell_class"].fillna("").values
side = m["side"].fillna("").values
ctype = m["cell_type"].fillna("").values

is_desc = (cc == "descending_neuron")
is_motor = np.array(["motor" in str(x) for x in cc])
leg = np.array(["leg_motor" in str(x) for x in cc])
wing = np.array(["wing_motor" in str(x) for x in cc])
neck = np.array(["neck_motor" in str(x) for x in cc])
sens = (m["super_class"].fillna("") == "sensory").values
asc = (m["super_class"].fillna("") == "ascending").values

desc = np.where(is_desc)[0].astype(np.int32)
motor_all = np.where(is_motor)[0].astype(np.int32)
leg_L = np.where(leg & (side == "left"))[0].astype(np.int32)
leg_R = np.where(leg & (side == "right"))[0].astype(np.int32)
wing_L = np.where(wing & (side == "left"))[0].astype(np.int32)
wing_R = np.where(wing & (side == "right"))[0].astype(np.int32)
neck_all = np.where(neck)[0].astype(np.int32)
sensory = np.where(sens)[0].astype(np.int32)
ascending = np.where(asc)[0].astype(np.int32)
print(f"desc={len(desc)} motor={len(motor_all)} leg={len(leg_L)}/{len(leg_R)} "
      f"wing={len(wing_L)}/{len(wing_R)} neck={len(neck_all)} sens={len(sensory)}", flush=True)

print("load edgelist...", flush=True)
e = pd.read_feather("manc_simple.feather")
pre_raw = e["pre"].values.astype(np.int64)
post_raw = e["post"].values.astype(np.int64)
cnt = e["count"].values.astype(np.float32)
del e
print(f"E_raw={len(pre_raw)} syn={float(cnt.sum()):.0f}", flush=True)

print("map ids...", flush=True)
pre = np.array([id2i.get(int(x), -1) for x in pre_raw], dtype=np.int32)
post = np.array([id2i.get(int(x), -1) for x in post_raw], dtype=np.int32)
del pre_raw, post_raw
ok = (pre >= 0) & (post >= 0)
print(f"mapped={int(ok.sum())}/{len(ok)}", flush=True)
pre, post, cnt = pre[ok], post[ok], cnt[ok]
sign = sign_pre[pre]
w = np.abs(cnt)

fan = np.zeros(M, dtype=np.float64)
np.add.at(fan, post, w.astype(float))
fan[fan == 0] = 1.0

np.savez_compressed("vnc_circuit.npz", pre=pre, post=post, weight=w, sign=sign,
                     fan=fan.astype(np.float32), N=np.array([M]),
                     desc=desc, motor_all=motor_all, leg_L=leg_L, leg_R=leg_R,
                     wing_L=wing_L, wing_R=wing_R, neck=neck_all,
                     sensory=sensory, ascending=ascending)
print("saved vnc_circuit.npz", flush=True)

# ---- bridge brain DESC -> VNC desc by exact DN type ----
print("bridge...", flush=True)
s = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
idx2fly = {i: int(f) for f, i in fly2idx.items()}
fly2type = dict(zip(s["root_id"].astype(int), s["cell_type"].astype(str)))
r = np.load("roles_full.npz")
brain_desc = [int(x) for x in r["DESC"]]
btype_of = {}
for d in brain_desc:
    btype_of[d] = fly2type.get(idx2fly.get(d, -1), "UNK")

vnc_type = np.array([str(x) for x in ctype])
vnc_by_type = {}
for i in desc:
    vnc_by_type.setdefault(vnc_type[int(i)], []).append(int(i))
brain_by_type = {}
for d in brain_desc:
    brain_by_type.setdefault(btype_of[d], []).append(int(d))

shared = sorted(set(brain_by_type) & set(vnc_by_type) - {"UNK", ""})
bp, vp, tn = [], [], []
for t in shared:
    for b in brain_by_type[t]:
        for v in vnc_by_type[t]:
            bp.append(b)
            vp.append(v)
            tn.append(t)
bp = np.array(bp, dtype=np.int32)
vp = np.array(vp, dtype=np.int32)
print(f"shared_types={len(shared)} pairs={len(bp)} "
      f"brain_cov={len(set(bp))}/{len(brain_desc)} vnc_cov={len(set(vp))}/{len(desc)}", flush=True)
np.savez_compressed("vnc_bridge.npz", brain=bp, vnc=vp,
                     n_shared=np.array([len(shared)]))
with open("vnc_bridge_types.txt", "w") as f:
    f.write("\n".join(shared))
print("saved vnc_bridge.npz + types", flush=True)
print("PASS-vnc-build", flush=True)
