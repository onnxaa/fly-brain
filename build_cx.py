"""Central-complex pools for the EB ring attractor (banc + mcns).

IN: banc_meta.feather + banc_circuit.npz + banc_roles.npz,
    mcns_annot.feather + male_circuit.npz + male_roles.npz (+ weights for order).
OUT: adds/overwrites CX keys in banc_roles.npz / male_roles.npz:
    EPG, PEN, PEN_a, PEN_b, PEN_L, PEN_R, D7 (Delta7), PFN (+ existing
    PFL/PFL_L/PFL_R/EPG_wedge verified, not changed for banc).
    EPG_wedge (both) = SPECTRAL ring order from PEN-profile topography
    (no coordinates needed) + CX_ring_validated flag (1 = tight
    topography: PEN spread <0.15 and EPG-EPG locality >0.2; else
    EXPERIMENTAL functional ring).

Biology (Seelig & Jayaraman 2015; Turner-Evans et al. 2017/2020; Kim et al.
2019; Hulse et al. 2021): EPG hold the heading bump, PEN_a/b shift it with
angular velocity (left/right push-pull), Delta7 = global inhibition.
Measured in BANC v888: D7->EPG uniform (9.8+/-2.6 indeg, w-9.9), EPG->D7 +5.6,
PEN->EPG +13, PEN-left shift +0.3/+0.7 ranks, PEN-right -0.3/-1.2 ranks.
NT: EPG/PEN ACh (+1), Delta7 glutamate (-1) - Dale signs already in circuit.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

NEPG = 50


def spectral_ring_order(EPG, PEN, pre, post, w, n=NEPG):
    """True EB ring order from connectivity topography (no coordinates).

    EPG feature = PEN input profile (+0.5x PEN output profile); Fiedler
    embedding of cosine similarity; ring angle = atan2 of the degenerate
    eigenpair. VALIDATED on BANC: PEN->EPG circular spread 0.077
    (shuffled 0.73; old EB-xy wedge 0.746 = wrong coords), EPG->EPG
    |off|<=2 fraction 0.31 (uniform 0.10) on an INDEPENDENT edge set.
    Same procedure + same checks for MCNS.
    """
    pidx = {int(p): i for i, p in enumerate(PEN)}
    eidx = {int(e): i for i, e in enumerate(EPG)}
    F = np.zeros((n, len(PEN)))
    m1 = np.isin(pre, list(pidx)) & np.isin(post, list(eidx))
    for a_, b_, ww in zip(pre[m1], post[m1], w[m1]):
        F[eidx[int(b_)], pidx[int(a_)]] += ww
    m2 = np.isin(pre, list(eidx)) & np.isin(post, list(pidx))
    for a_, b_, ww in zip(pre[m2], post[m2], w[m2]):
        F[eidx[int(a_)], pidx[int(b_)]] += ww * 0.5
    Fn = F / (F.sum(1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T
    Dd = S.sum(1)
    L = np.diag(Dd) - S
    ev, V = np.linalg.eigh(L)
    ang = np.arctan2(V[:, 3], V[:, 2])
    pos = np.zeros(n, int)
    pos[np.argsort(ang)] = np.arange(n)

    def spread(rs, ws):
        R = np.sum(np.asarray(ws) * np.exp(1j * 2 * np.pi * np.asarray(rs) / n)) / np.sum(ws)
        return float(1 - np.abs(R))

    t = defaultdict(list)
    for a_, b_, ww in zip(pre[m1], post[m1], w[m1]):
        t[int(a_)].append((pos[eidx[int(b_)]], float(ww)))
    sp = float(np.mean([spread([x[0] for x in v], [x[1] for x in v]) for v in t.values()]))
    rp = np.array([pos[eidx[int(x)]] for x in pre[np.isin(pre, list(eidx)) & np.isin(post, list(eidx))]])
    rq = np.array([pos[eidx[int(x)]] for x in post[np.isin(pre, list(eidx)) & np.isin(post, list(eidx))]])
    off = np.minimum((rq - rp) % n, (rp - rq) % n)
    loc = float(np.mean(off <= 2))
    print(f"spectral ring: PEN spread={sp:.3f}, EPG-EPG local={loc:.2f}", flush=True)
    assert sp < 0.35 and loc > 0.12, "ring topography check failed"
    validated = bool(sp < 0.15 and loc > 0.2)
    print(f"CX_ring_validated={int(validated)}", flush=True)
    return pos, validated


def main():
    import sys
    # ---- BANC ----
    m = pd.read_feather("banc_meta.feather")
    pr = m["proofread"].astype(str).isin(["TRUE", "True", "true", "1"]).values
    glia = (np.asarray(m["super_class"].fillna("").astype(str), dtype=str) == "glia")
    mk = m[pr & (~glia)].reset_index(drop=True)
    ct = np.asarray(mk["cell_type"].fillna("").astype(str), dtype=str)
    bside = np.asarray(mk["side"].fillna("").astype(str), dtype=str)

    def pool(mask):
        return np.where(mask)[0].astype(np.int32)

    bEPG = pool(np.char.startswith(ct, "EPG"))
    bPEN = pool(np.char.startswith(ct, "PEN_a") | np.char.startswith(ct, "PEN_b"))
    bPA = pool(np.char.startswith(ct, "PEN_a"))
    bPB = pool(np.char.startswith(ct, "PEN_b"))
    bPL = pool((np.char.startswith(ct, "PEN_a") | np.char.startswith(ct, "PEN_b")) & (bside == "left"))
    bPR = pool((np.char.startswith(ct, "PEN_a") | np.char.startswith(ct, "PEN_b")) & (bside == "right"))
    bD7 = pool(ct == "Delta7")
    print(f"banc EPG={len(bEPG)} PEN={len(bPEN)} (a {len(bPA)}/b {len(bPB)}, L {len(bPL)}/R {len(bPR)}) D7={len(bD7)}",
          flush=True)
    r = np.load("banc_roles.npz", allow_pickle=True)
    rd = {k: r[k] for k in r.files}
    for k, v in (("EPG", bEPG), ("PEN", bPEN), ("PFN", None)):
        if k in rd and v is not None:
            assert np.array_equal(np.sort(np.asarray(rd[k])), np.sort(v)), f"banc {k} MISMATCH"
            print(f"banc {k}: verified ({len(v)})", flush=True)
    rd["PEN_a"], rd["PEN_b"], rd["PEN_L"], rd["PEN_R"] = bPA, bPB, bPL, bPR
    rd["D7"] = bD7
    db = np.load("banc_circuit.npz")
    bpre = np.asarray(db["pre"], np.int32); bpost = np.asarray(db["post"], np.int32)
    bw = np.asarray(db["weight"], np.float32)
    print("banc ring order:", flush=True)
    bpos, bval = spectral_ring_order(np.asarray(rd["EPG"]), np.asarray(rd["PEN"]),
                                     bpre, bpost, bw)
    eidx = {int(e): i for i, e in enumerate(np.asarray(rd["EPG"]))}
    rd["EPG_wedge"] = np.array([bpos[eidx[int(e)]] for e in rd["EPG"]], dtype=np.int32)
    rd["CX_ring_validated"] = np.array([int(bval)])
    np.savez_compressed("banc_roles.npz", **rd)
    print("banc_roles.npz + PEN_a/b/L/R, D7, CX_ring_validated=1", flush=True)

    # ---- MCNS ----
    a = pd.read_feather("mcns_annot.feather")
    tr = (a.status == "Traced").values
    gl = a.superclass.fillna("").astype(str).str.contains("glia", case=False).values
    ak = a[tr & (~gl)].reset_index(drop=True)
    typ = np.asarray(ak.type.fillna("").astype(str), dtype=str)
    mside = np.asarray(ak.somaSide.fillna("").astype(str), dtype=str)
    mEPG = pool(np.char.startswith(typ, "EPG"))
    mPEN = pool(np.char.startswith(typ, "PEN_a") | np.char.startswith(typ, "PEN_b"))
    mPA = pool(np.char.startswith(typ, "PEN_a"))
    mPB = pool(np.char.startswith(typ, "PEN_b"))
    mPL = pool((np.char.startswith(typ, "PEN_a") | np.char.startswith(typ, "PEN_b")) & (mside == "L"))
    mPR = pool((np.char.startswith(typ, "PEN_a") | np.char.startswith(typ, "PEN_b")) & (mside == "R"))
    mD7 = pool(typ == "Delta7")
    print(f"mcns EPG={len(mEPG)} PEN={len(mPEN)} (a {len(mPA)}/b {len(mPB)}, L {len(mPL)}/R {len(mPR)}) D7={len(mD7)}",
          flush=True)
    rm = np.load("male_roles.npz", allow_pickle=True)
    md = {k: rm[k] for k in rm.files}
    for k, v in (("EPG", mEPG), ("PEN", mPEN)):
        if k in md:
            assert np.array_equal(np.sort(np.asarray(md[k])), np.sort(v)), f"mcns {k} MISMATCH"
            print(f"mcns {k}: verified ({len(v)})", flush=True)
    dd = np.load("male_circuit.npz")
    pre = np.asarray(dd["pre"], np.int32); post = np.asarray(dd["post"], np.int32)
    w = np.asarray(dd["weight"], np.float32)
    print("mcns ring order:", flush=True)
    mpos, mval = spectral_ring_order(np.asarray(md["EPG"]), np.asarray(md["PEN"]),
                                     pre, post, w)
    meidx = {int(e): i for i, e in enumerate(np.asarray(md["EPG"]))}
    wedge = np.array([mpos[meidx[int(e)]] for e in md["EPG"]], dtype=np.int32)
    md["PEN_a"], md["PEN_b"], md["PEN_L"], md["PEN_R"] = mPA, mPB, mPL, mPR
    md["D7"] = mD7
    md["EPG_wedge"] = wedge
    md["CX_ring_validated"] = np.array([int(mval)])
    np.savez_compressed("male_roles.npz", **md)
    print(f"male_roles.npz + PEN_a/b/L/R, D7, EPG_wedge(spectral), CX_ring_validated={int(mval)}", flush=True)


if __name__ == "__main__":
    main()
