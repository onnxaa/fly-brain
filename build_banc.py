"""Build female whole-CNS circuit from BANC v888 (Bates/Phelps/Kim/Yang, Nature 2026).

IN: banc_meta.feather (188k) + banc_edge_v2.feather (11.75M rows).
OUT: banc_circuit.npz (pre/post/weight/sign/fan, count>=5 Codex parity,
       autapses dropped) + banc_roles.npz (pools) + banc_door.npz (6 DoOR
       odors via ORN_<glom> suffix, same consensus as door_real.py).
Neuron set: proofread + non-glia. Signs from NT: ACh=+1, GABA/glutamate/
  histamine=-1 (central), DA/OA/5HT/tyramine/unknown=+1 (modulatory, protocol).
Valence (approach/avoid) ported from FAFB via fafb_match (FAFB root ids).
Streams edges in batches (low RAM). RULE: frozen topology/Dale from BANC.
"""
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

print("meta...", flush=True)
m = pd.read_feather("banc_meta.feather")
pr = m["proofread"].astype(str).isin(["TRUE", "True", "true", "1"]).values
glia = (np.asarray(m["super_class"].fillna("").astype(str), dtype=str) == "glia")
keep = pr & (~glia)
mk = m[keep].reset_index(drop=True)
K = len(mk)
print(f"kept {K}/{len(m)} (proofread non-glia)", flush=True)
ids = np.asarray(mk["banc_888_id"].astype(str), dtype=str)
order = np.argsort(ids)
sids = ids[order]
# order[sorted_pos] = mk position (= local idx): searchsorted -> order -> local

nt = mk["neurotransmitter_predicted"].fillna("unknown").astype(str).str.lower().values
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 1.0, "octopamine": 1.0, "serotonin": 1.0, "tyramine": 1.0,
        "unknown": 1.0, "unclear": 1.0}
nt_sign = np.array([SIGN.get(x, 1.0) for x in nt], dtype=np.float32)
region = np.asarray(mk["region"].fillna("").astype(str), dtype=str)
side = np.asarray(mk["side"].fillna("").astype(str), dtype=str)
sc = np.asarray(mk["super_class"].fillna("").astype(str), dtype=str)
cc = np.asarray(mk["cell_class"].fillna("").astype(str), dtype=str)
ct = np.asarray(mk["cell_type"].fillna("").astype(str), dtype=str)


def pool(mask):
    return np.where(mask)[0].astype(np.int32)


KC = pool(np.char.startswith(ct.astype(str), "KC"))
MBON = pool(np.char.startswith(ct.astype(str), "MBON"))
DAN = pool(cc == "mushroom_body_dopaminergic_neuron")
dan_pam = pool((cc == "mushroom_body_dopaminergic_neuron") & np.char.startswith(ct.astype(str), "PAM"))
dan_ppl = pool((cc == "mushroom_body_dopaminergic_neuron") & np.char.startswith(ct.astype(str), "PPL"))
ORN = pool(cc == "olfactory_receptor_neuron")
DESC = pool(sc == "descending")
DESC_L = pool((sc == "descending") & (side == "left"))
DESC_R = pool((sc == "descending") & (side == "right"))
MOTOR = pool(sc == "motor")
leg_L = pool((cc == "leg_motor_neuron") & (side == "left"))
leg_R = pool((cc == "leg_motor_neuron") & (side == "right"))
wing = pool(cc == "wing_motor_neuron")
neck = pool(cc == "neck_motor_neuron")
ASC = pool(sc == "ascending")
SENS = pool(sc == "sensory")
print(f"KC={len(KC)} MBON={len(MBON)} DAN={len(DAN)} (pam {len(dan_pam)}/ppl {len(dan_ppl)}) "
      f"ORN={len(ORN)} DESC={len(DESC)} MOTOR={len(MOTOR)}", flush=True)

# valence via fafb_match -> FAFB global idx
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
mb = np.load("mb_circuit.npz")
mb_fly = mb["flywire_ids"]
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
g = np.load("mb_groups_v3.npz")
app_fafb = set(int(loc2glob[a]) for a in g["approach"])
avo_fafb = set(int(loc2glob[a]) for a in g["avoid"])
fafb_root_of_idx = {i: int(f) for i, f in enumerate(fly_ids)}
fm = np.asarray(mk["fafb_match"].fillna("").astype(str), dtype=str)
app_ids = {str(int(fly_ids[int(loc2glob[a])])) for a in g["approach"]}
avo_ids = {str(int(fly_ids[int(loc2glob[a])])) for a in g["avoid"]}
approach = pool(np.isin(fm, list(app_ids)) & np.isin(np.arange(K), MBON))
avoid = pool(np.isin(fm, list(avo_ids)) & np.isin(np.arange(K), MBON))
print(f"valence ported: approach {len(approach)}/71 FAFB, avoid {len(avoid)}/25 FAFB", flush=True)

# ORN glomeruli from type suffix
orn_glom = np.array([str(t)[4:] if str(t).startswith("ORN_") else "" for t in ct[ORN]])
banc_orn_idx = ORN  # local indices

print("edges (stream)...", flush=True)
P, Q, W = [], [], []
nraw = nkept = nge5 = 0
with ipc.open_file("banc_edge_v2.feather") as f:
    nb = f.num_record_batches
    for bi in range(nb):
        b = f.get_batch(bi)
        d = b.to_pandas()
        nraw += len(d)
        nge5 += len(d)
        d = d[d["pre"] != d["post"]]
        pv = np.asarray(d["pre"].astype(str), dtype=str)
        qv = np.asarray(d["post"].astype(str), dtype=str)
        a = np.searchsorted(sids, pv)
        bb = np.searchsorted(sids, qv)
        ok = (a < K) & (bb < K) & (sids[np.clip(a, 0, K - 1)] == pv) & \
             (sids[np.clip(bb, 0, K - 1)] == qv)
        a, bb = order[a[ok]], order[bb[ok]]
        P.append(a.astype(np.int32))
        Q.append(bb.astype(np.int32))
        W.append(d["count"].values[ok].astype(np.float32))
        nkept += int(ok.sum())
        del d, a, bb
        if (bi + 1) % 20 == 0:
            print(f" batch {bi+1}/{nb} kept={nkept}", flush=True)
pre = np.concatenate(P)
post = np.concatenate(Q)
cnt = np.concatenate(W)
del P, Q, W
print(f"E_raw={nraw} kept={len(pre)} (no count cut: v2 size>=5 at detection)", flush=True)
sign = nt_sign[pre]
w = cnt  # |w| = synapse count
fan = np.zeros(K, dtype=np.float64)
np.add.at(fan, post, w.astype(float))
fan[fan == 0] = 1.0
np.savez_compressed("banc_circuit.npz", pre=pre, post=post, weight=w, sign=sign,
                     fan=fan.astype(np.float32), N=np.array([K]))
print("saved banc_circuit.npz", flush=True)
del pre, post, cnt, w, sign

# intact neck connective: edges crossing brain<->VNC regions
is_vnc = (region == "ventral_nerve_cord").astype(np.int8)
np.savez_compressed("banc_roles.npz", KC=KC, MBON=MBON, DAN=DAN, dan_pam=dan_pam,
                     dan_ppl=dan_ppl, ORN=banc_orn_idx, ORN_glom=orn_glom,
                     DESC=DESC, DESC_L=DESC_L, DESC_R=DESC_R, MOTOR=MOTOR,
                     leg_L=leg_L, leg_R=leg_R, wing=wing, neck=neck,
                     ASC=ASC, SENS=SENS, approach=approach, avoid=avoid,
                     side=np.array(side), region=np.array(region), K=np.array([K]))
print("saved banc_roles.npz", flush=True)

# DoOR: same consensus as door_real.py, glomerulus from ORN_ suffix
M = pd.read_csv("door_matrix.csv", sep=";", index_col=0)
mp = pd.read_csv("door_mappings.csv", sep=";")
unit2glom = {}
for _, r in mp.iterrows():
    gg = str(r["glomerulus"])
    if gg in ("?", "", "nan") or pd.isna(r["glomerulus"]):
        continue
    for col in ["receptor", "code", "code.OSN", "OSN"]:
        u = str(r[col])
        if u not in ("?", "", "nan") and not pd.isna(r[col]):
            unit2glom.setdefault(u, gg)
od = pd.read_csv("door_odor.csv", sep=";").set_index("InChIKey")
KEYS = {}
for name, cas in [("geosmin", "16423-19-1"), ("co2", "124-38-9"), ("hexanone3", "589-38-8"),
                  ("methyl_salicylate", "119-36-8"), ("butanedione", "431-03-8"),
                  ("ethyl_hexanoate", "123-66-0")]:
    hit = od[od["CAS"].astype(str) == cas]
    KEYS[name] = hit.index[0]
out = {}
for name, key in KEYS.items():
    row = M.loc[key].astype(float)
    gprof = {}
    for unit, val in row.items():
        if pd.isna(val):
            continue
        gg = unit2glom.get(str(unit))
        if gg:
            gprof.setdefault(gg, []).append(val)
    gmean = {gg: float(np.mean(v)) for gg, v in gprof.items()}
    mx = max([abs(v) for v in gmean.values()] + [1e-9])
    vals = np.array([float(np.clip(gmean.get(gl, 0.0) / mx, -1, 1)) for gl in orn_glom],
                    dtype=np.float32)
    out[name + "_idx"] = banc_orn_idx.astype(np.int32)
    out[name + "_val"] = vals
np.savez("banc_door.npz", **out)
print("saved banc_door.npz", flush=True)
print("PASS-banc-build", flush=True)
