"""Real R front-end for BANC + MCNS (see FAFB roles_full R/R_cx/R_cy recipe).

BANC: R7/R8 proofread neurons (no R1-6 annotated), RF from meta `position`
  x/y (stable point ~ ommatidium; rank-normalized like FAFB pos_x/pos_y).
MCNS: R1-6/R7/R8 with non-null somaLocation, RF from soma x/y rank-norm.
Side (L/R) kept for laterality. Updates banc_roles.npz / male_roles.npz
with R, R_cx, R_cy, R_side. RULE: positions from data, rank-norm is protocol
(same as FAFB).
"""
import numpy as np
import pandas as pd
import re

# ---------- BANC ----------
m = pd.read_feather("banc_meta.feather")
ct = np.asarray(m.cell_type.fillna("").astype(str), dtype=str)
pr = m.proofread.astype(str).values == "TRUE"
ng = np.asarray(m.super_class.fillna("").astype(str), dtype=str) != "glia"
keep = pr & ng
kids = np.asarray(m.banc_888_id.astype(str), dtype=str)[keep]
kpos = {v: i for i, v in enumerate(kids)}
isR = np.array([bool(re.match(r"^R[78]", t)) for t in ct])
rrows = np.where(keep & isR)[0]
pos = m.position.fillna("").astype(str).values[rrows]
XY = []
ids = []
for r, p in zip(rrows, pos):
    try:
        x, y, _ = [float(v) for v in str(p).split(",")]
    except Exception:
        continue
    ids.append(str(m.banc_888_id.astype(str).values[r]))
    XY.append((x, y))
XY = np.array(XY)
R = np.array([kpos[i] for i in ids], dtype=np.int32)
ox = np.argsort(XY[:, 0], kind="stable")
oy = np.argsort(XY[:, 1], kind="stable")
cx = np.empty(len(R)); cy = np.empty(len(R))
cx[ox] = np.arange(len(R)) / max(1, len(R) - 1)
cy[oy] = np.arange(len(R)) / max(1, len(R) - 1)
rside = np.asarray(m.side.fillna("").astype(str), dtype=str)[rrows][:len(R)]
# align rside to kept ids (rrows filtered by parse success)
rside = []
for r in rrows:
    p = str(m.position.fillna("").astype(str).values[r])
    try:
        [float(v) for v in p.split(",")]
        rside.append(str(m.side.fillna("").astype(str).values[r]))
    except Exception:
        pass
rside = np.array(rside)
print(f"BANC R: n={len(R)} (R7/R8 only, no R1-6 annotated)")
rb = dict(np.load("banc_roles.npz", allow_pickle=True))
rb["R"] = R.astype(np.int32)
rb["R_cx"] = cx.astype(np.float32)
rb["R_cy"] = cy.astype(np.float32)
rb["R_side"] = rside
np.savez_compressed("banc_roles.npz", **rb)
# L1/L2 luminance proxy (R1-6 absent in v888): column positions as RF.
mk_ids = np.asarray(m.banc_888_id.astype(str), dtype=str)
kept_pos = {v: i for i, v in enumerate(np.asarray(m.banc_888_id.astype(str), dtype=str)[keep])}
Lpools = {}
for t in ["L1", "L2"]:
    sel = np.where(keep & (ct == t))[0]
    XY, ids = [], []
    for r in sel:
        try:
            x, y, _ = [float(v) for v in str(m.position.fillna("").astype(str).values[r]).split(",")]
            XY.append((x, y)); ids.append(str(m.banc_888_id.astype(str).values[r]))
        except Exception:
            pass
    XY = np.array(XY)
    Li = np.array([kept_pos[i] for i in ids], dtype=np.int32)
    ox = np.argsort(XY[:, 0], kind="stable"); oy = np.argsort(XY[:, 1], kind="stable")
    cx = np.empty(len(Li)); cy = np.empty(len(Li))
    cx[ox] = np.arange(len(Li)) / max(1, len(Li) - 1)
    cy[oy] = np.arange(len(Li)) / max(1, len(Li) - 1)
    Lpools[t] = (Li, cx.astype(np.float32), cy.astype(np.float32))
    print(f"BANC {t}: n={len(Li)}")
rb = dict(np.load("banc_roles.npz", allow_pickle=True))
rb["L1"] = Lpools["L1"][0]; rb["L1_cx"] = Lpools["L1"][1]; rb["L1_cy"] = Lpools["L1"][2]
rb["L2"] = Lpools["L2"][0]; rb["L2_cx"] = Lpools["L2"][1]; rb["L2_cy"] = Lpools["L2"][2]
np.savez_compressed("banc_roles.npz", **rb)
print("patched banc_roles.npz (L proxy)")

# ---------- MCNS ----------
# No coordinates in v1.0 flat files (somaLocation 28/6091) — honest eye split
# only (rootSide L/R). No fake per-neuron RF (that would break pure).
a = pd.read_feather("mcns_annot.feather")
t = np.asarray(a.type.fillna("").astype(str), dtype=str)
tr = (a.status == "Traced").values
gl = a.superclass.fillna("").astype(str).str.contains("glia", case=False).values
mk = a[tr & ~gl].reset_index(drop=True)
bids = mk.bodyId.values.astype(np.int64)
mt = np.asarray(mk.type.fillna("").astype(str), dtype=str)
isR = np.array([bool(re.match(r"^(R[178]|R[178][a-z_])", x)) for x in mt])
R2 = np.where(isR)[0].astype(np.int32)
S2 = np.asarray(mk.rootSide.fillna("").astype(str), dtype=str)[R2]
print(f"MCNS R: n={len(R2)} eye-split only (L={(S2=='L').sum()}/R={(S2=='R').sum()})")
rm = dict(np.load("male_roles.npz", allow_pickle=True))
# ---- MCNS homology retinotopy (BANC malecns_match -> BANC position) ----
# 404/4107 MCNS R match BANC R7/R8 (100% type-consistent). RF = rank-norm of
# matched BANC coords among themselves. Unmatched stay eye-mean (no fabrication).
bm=pd.read_feather("banc_meta.feather", columns=["banc_888_id","cell_type","position","malecns_match"])
bct=np.asarray(bm.cell_type.fillna("").astype(str),dtype=str)
bisR=np.array([bool(re.match(r"^R[78]",x)) for x in bct])
bpos_map={}
for i in np.where(bisR)[0]:
    mm=str(bm.malecns_match.fillna("").astype(str).values[i])
    if not mm: continue
    try:
        x,y,_=[float(v) for v in str(bm.position.fillna("").astype(str).values[i]).split(",")]
        bpos_map.setdefault(mm,[]).append((x,y))
    except Exception: pass
bpos_mean={k:(float(np.mean([p[0] for p in v])),float(np.mean([p[1] for p in v]))) for k,v in bpos_map.items()}
mbids=np.asarray(mk.bodyId.values,dtype=np.int64)
h_ids, h_xy=[],[]
for li in R2:
    mm=str(int(mbids[li]))
    if mm in bpos_mean:
        h_ids.append(int(li)); h_xy.append(bpos_mean[mm])
h_ids=np.array(h_ids,np.int32); h_xy=np.array(h_xy)
ox=np.argsort(h_xy[:,0],kind="stable"); oy=np.argsort(h_xy[:,1],kind="stable")
hcx=np.empty(len(h_ids)); hcy=np.empty(len(h_ids))
hcx[ox]=np.arange(len(h_ids))/max(1,len(h_ids)-1)
hcy[oy]=np.arange(len(h_ids))/max(1,len(h_ids)-1)
print(f"MCNS homology RF: {len(h_ids)}/{len(R2)} anchored")
rm = dict(np.load("male_roles.npz", allow_pickle=True))
rm["R"] = R2.astype(np.int32)
rm["R_side"] = np.array(S2)
rm["Rret"] = h_ids.astype(np.int32)
rm["Rret_cx"] = hcx.astype(np.float32)
rm["Rret_cy"] = hcy.astype(np.float32)
np.savez_compressed("male_roles.npz", **rm)
print("patched male_roles.npz")
print("PASS-rf-build", flush=True)
