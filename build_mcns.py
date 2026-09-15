"""Build male whole-CNS circuit from MCNS v1.0 (Berg et al., Cell 2026).

IN: mcns_annot.feather (211k) + mcns_nt.feather (1.8M synapse rows) +
    mcns_weights.feather (body_pre/body_post/weight int64).
OUT: male_circuit.npz (pre/post/weight/sign/fan, autapses dropped) +
     male_roles.npz (pools) + male_door.npz (6 DoOR odors via ORN_<glom>).
Neuron set: status Traced, superclass != glia. Signs: per-body NT consensus
  (non-unclear majority; fallback celltype_predicted_nt; ACh=+1, GABA/glu/
  histamine=-1, monoamines/unknown=+1 protocol). Valence ported FAFB->MCNS
  by MBON type name (supp1.tsv). RULE: frozen topology/Dale from MCNS.
"""
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

print("annot...", flush=True)
a = pd.read_feather("mcns_annot.feather")
tr = (a.status == "Traced").values
gl = a.superclass.fillna("").astype(str).str.contains("glia", case=False).values
keep = tr & (~gl)
mk = a[keep].reset_index(drop=True)
K = len(mk)
print(f"kept {K}/{len(a)} (traced non-glia)", flush=True)
bids = mk.bodyId.values.astype(np.int64)
order = np.argsort(bids)
sbids = bids[order]

typ = np.asarray(mk.type.fillna("").astype(str), dtype=str)
sc = np.asarray(mk.superclass.fillna("").astype(str), dtype=str)
side = np.asarray(mk.somaSide.fillna("").astype(str), dtype=str)


def pool(mask):
    return np.where(mask)[0].astype(np.int32)


KC = pool(np.char.startswith(typ, "KC"))
MBON = pool(np.char.startswith(typ, "MBON"))
pam = pool(np.char.startswith(typ, "PAM"))
ppl = pool(np.char.startswith(typ, "PPL"))
DAN = pool(np.char.startswith(typ, "PAM") | np.char.startswith(typ, "PPL") |
           (typ == "DAN") | np.char.startswith(typ, "MeVPaMe"))
ORN = pool(np.char.startswith(typ, "ORN"))
DESC = pool(sc == "descending_neuron")
DESC_L = pool((sc == "descending_neuron") & (side == "L"))
DESC_R = pool((sc == "descending_neuron") & (side == "R"))
MOTOR = pool(sc == "vnc_motor")
exitn = np.asarray(mk.exitNerve.fillna("").astype(str), dtype=str)
_leg = np.isin(exitn, ["ProLN", "MesoLN", "MetaLN"])
_wing = np.array([bool(__import__("re").search(r"DLM|DVM|ergopleural|teering|haltere", t, __import__("re").I)) for t in typ])
leg_L = pool((sc == "vnc_motor") & (side == "L") & _leg)
leg_R = pool((sc == "vnc_motor") & (side == "R") & _leg)
wing = pool((sc == "vnc_motor") & _wing)
neck = pool((sc == "vnc_motor") & (exitn == "CvN"))
SENS = pool(np.char.endswith(sc, "sensory") | (sc == "sensory_ascending"))
SENS_L = pool(((np.char.endswith(sc, "sensory")) | (sc == "sensory_ascending")) & (side == "L"))
SENS_R = pool(((np.char.endswith(sc, "sensory")) | (sc == "sensory_ascending")) & (side == "R"))
print(f"KC={len(KC)} MBON={len(MBON)} DAN={len(DAN)} (pam {len(pam)}/ppl {len(ppl)}) "
      f"ORN={len(ORN)} DESC={len(DESC)} MOTOR={len(MOTOR)}", flush=True)
print("motor classes:", np.unique(mclass[MOTOR])[:12].tolist(), flush=True)

# valence: FAFB MBON names (supp1) -> MCNS type names
s = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
fly2type = dict(zip(s.root_id.astype(int), s.cell_type.astype(str)))
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
mb = np.load("mb_circuit.npz")
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb["flywire_ids"]], dtype=np.int32)
g = np.load("mb_groups_v3.npz")
app_types = {str(fly2type.get(int(fly_ids[int(loc2glob[x])]), "")) for x in g["approach"]}
avo_types = {str(fly2type.get(int(fly_ids[int(loc2glob[x])]), "")) for x in g["avoid"]}
approach = pool(np.isin(typ, list(app_types)) & np.isin(np.arange(K), MBON))
avoid = pool(np.isin(typ, list(avo_types)) & np.isin(np.arange(K), MBON))
print(f"valence: approach {len(approach)} avoid {len(avoid)}", flush=True)

# per-body NT consensus
print("nt consensus...", flush=True)
n = pd.read_feather("mcns_nt.feather", columns=["body", "predicted_nt"])
n = n[~n.predicted_nt.isin(["unclear"])]
maj = n.groupby("body").predicted_nt.agg(lambda x: x.value_counts().index[0])
majd = {int(k): str(v).lower() for k, v in maj.items()}
nt2 = pd.read_feather("mcns_nt.feather", columns=["cell_type", "celltype_predicted_nt"])
ctmap2 = (nt2.groupby("cell_type").celltype_predicted_nt.agg(
    lambda x: x.value_counts().index[0]).to_dict())
ctmap2 = {str(k): str(v).lower() for k, v in ctmap2.items()}
mct = np.asarray(mk.type.fillna("").astype(str), dtype=str)
ctmap = {}
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 1.0, "octopamine": 1.0, "serotonin": 1.0, "tyramine": 1.0,
        "unknown": 1.0, "unclear": 1.0}
pre_nt = np.array([SIGN.get(majd.get(int(bb), ctmap2.get(mct[i], "unknown")), 1.0)
                   for i, bb in enumerate(mk.bodyId.values.astype(np.int64))],
                  dtype=np.float32)
print("NT signs: +1:", int((pre_nt > 0).sum()), "-1:", int((pre_nt < 0).sum()), flush=True)

print("edges (stream)...", flush=True)
P, Q, W = [], [], []
nraw = nkept = 0
with ipc.open_file("mcns_weights.feather") as f:
    nb = f.num_record_batches
    for bi in range(nb):
        b = f.get_batch(bi).to_pandas()
        nraw += len(b)
        b = b[b.body_pre != b.body_post]
        pa = b.body_pre.values.astype(np.int64)
        qa = b.body_post.values.astype(np.int64)
        ia = np.searchsorted(sbids, pa)
        ib = np.searchsorted(sbids, qa)
        ok = (ia < K) & (ib < K) & (sbids[np.clip(ia, 0, K - 1)] == pa) & \
             (sbids[np.clip(ib, 0, K - 1)] == qa)
        P.append(order[ia[ok]].astype(np.int32))
        Q.append(order[ib[ok]].astype(np.int32))
        W.append(b.weight.values[ok].astype(np.float32))
        nkept += int(ok.sum())
        del b, pa, qa, ia, ib, ok
        if (bi + 1) % 20 == 0:
            print(f" batch {bi+1}/{nb} kept={nkept}", flush=True)
pre = np.concatenate(P)
post = np.concatenate(Q)
cnt = np.concatenate(W)
del P, Q, W
print(f"E_raw={nraw} kept={len(pre)}", flush=True)
sign = pre_nt[pre]
fan = np.zeros(K, dtype=np.float64)
np.add.at(fan, post, cnt.astype(float))
fan[fan == 0] = 1.0
np.savez_compressed("male_circuit.npz", pre=pre, post=post, weight=cnt, sign=sign,
                     fan=fan.astype(np.float32), N=np.array([K]))
print("saved male_circuit.npz", flush=True)
del pre, post, cnt, sign

orn_glom = np.array([t[4:] if t.startswith("ORN_") else "" for t in typ[ORN]])
np.savez_compressed("male_roles.npz", KC=KC, MBON=MBON, DAN=DAN, dan_pam=pam,
                     dan_ppl=ppl, ORN=ORN, ORN_glom=orn_glom, DESC=DESC,
                     DESC_L=DESC_L, DESC_R=DESC_R, MOTOR=MOTOR, leg_L=leg_L,
                     leg_R=leg_R, wing=wing, neck=neck, SENS=SENS,
                     SENS_L=SENS_L, SENS_R=SENS_R, approach=approach,
                     avoid=avoid, side=side, K=np.array([K]))
print("saved male_roles.npz", flush=True)

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
    out[name + "_idx"] = ORN.astype(np.int32)
    out[name + "_val"] = np.array([float(np.clip(gmean.get(gl, 0.0) / mx, -1, 1))
                                   for gl in orn_glom], dtype=np.float32)
np.savez("male_door.npz", **out)
print("saved male_door.npz", flush=True)
print("PASS-mcns-build", flush=True)
