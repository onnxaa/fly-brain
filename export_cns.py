"""Export BANC + MCNS circuits to flat little-endian binaries for fly_cpp.

Reads banc_circuit/roles/door + male_circuit/roles/door (see build_banc.py,
build_mcns.py) and writes CSR-direct layout (stable sort by post =>
bit-identical FP order with Python np.add.at) + pools + door + manifest
entries into ../fly_cpp/data/. Mode prefix: banc_* / male_*.
"""
import json
import os
import numpy as np

SRC = "."
DST = "../fly_cpp/data"
os.makedirs(DST, exist_ok=True)
man = json.load(open(f"{DST}/manifest.json")) if os.path.exists(f"{DST}/manifest.json") else {}


def w(name, arr):
    arr = np.ascontiguousarray(arr)
    ext = {"int32": "i32", "float32": "f32", "int64": "i64",
           "uint8": "u8"}[str(arr.dtype)]
    fn = f"{DST}/{name}.{ext}"
    arr.tofile(fn)
    man[name] = {"file": os.path.basename(fn), "n": int(arr.size),
                 "dtype": str(arr.dtype)}
    print(f"{name}: n={arr.size} dtype={arr.dtype}", flush=True)


ODORS = ("geosmin", "co2", "hexanone3", "methyl_salicylate",
         "butanedione", "ethyl_hexanoate")

for mode, circ, roles in (("banc", "banc_circuit.npz", "banc_roles.npz"),
                          ("male", "male_circuit.npz", "male_roles.npz")):
    d = np.load(f"{SRC}/{circ}")
    pre = np.asarray(d["pre"]).astype(np.int32)
    post = np.asarray(d["post"]).astype(np.int32)
    sw = (np.asarray(d["weight"]).astype(np.float32)
          * np.asarray(d["sign"]).astype(np.float32))
    o = np.argsort(post, kind="stable")
    w(f"{mode}_pre", pre[o])
    w(f"{mode}_post", post[o])
    w(f"{mode}_w", sw[o])  # signed; C++ splits sign/|w|
    w(f"{mode}_fan", np.asarray(d["fan"]).astype(np.float32))
    r = np.load(f"{SRC}/{roles}", allow_pickle=True)
    for k, nk in (("KC", "KC"), ("MBON", "MBON"), ("approach", "approach"),
                  ("avoid", "avoid"), ("dan_pam", "dan_pam"),
                  ("dan_ppl", "dan_ppl"), ("ORN", "ORN"),
                  ("DESC_L", "DESC_L"), ("DESC_R", "DESC_R"),
                  ("MOTOR", "MOTOR"), ("leg_L", "leg_L"), ("leg_R", "leg_R"),
                  ("wing", "wing"), ("neck", "neck")):
        if k in r:
            w(f"{mode}_{nk}", np.asarray(r[k]).astype(np.int32))
    # MECH: banc SENS / male SENS + laterality by side
    side = np.asarray(r["side"]).astype(str)
    for pool_key, side_key in (("SENS", None),):
        if pool_key in r:
            pool = np.asarray(r[pool_key]).astype(np.int32)
            w(f"{mode}_MECH", pool)
            w(f"{mode}_MECH_L", pool[side[pool] == ("left" if mode == "banc" else "L")])
            w(f"{mode}_MECH_R", pool[side[pool] == ("right" if mode == "banc" else "R")])
    orn = np.asarray(r["ORN"]).astype(np.int32)
    w(f"{mode}_ORN_L", orn[side[orn] == ("left" if mode == "banc" else "L")])
    w(f"{mode}_ORN_R", orn[side[orn] == ("right" if mode == "banc" else "R")])
    for ck in ("CLOCK_M", "CLOCK_E", "GRN"):
        if ck in r:
            w(f"{mode}_{ck.lower()}", np.asarray(r[ck]).astype(np.int32))
    if "GRN" in r:
        grn = np.asarray(r["GRN"]).astype(np.int32)
        for t in ("sugar", "bitter", "water", "ir94e"):
            # single GRN pool, no quality split (see API docs)
            w(f"{mode}_taste_{t}", grn)
    if "R" in r:
        rr = np.asarray(r["R"]).astype(np.int32)
        w(f"{mode}_R", rr)
        if "R_cx" in r:
            w(f"{mode}_R_cx", np.asarray(r["R_cx"]).astype(np.float32))
            w(f"{mode}_R_cy", np.asarray(r["R_cy"]).astype(np.float32))
        if "R_side" in r:
            rs = np.asarray(r["R_side"]).astype(str)
            tag = "left" if mode == "banc" else "L"
            tagR = "right" if mode == "banc" else "R"
            w(f"{mode}_R_L", rr[rs == tag])
            w(f"{mode}_R_R", rr[rs == tagR])
    for lk in ("L1", "L2", "L1_cx", "L1_cy", "L2_cx", "L2_cy"):
        if lk in r:
            w(f"{mode}_{lk}", np.asarray(r[lk]).astype(np.int32 if not lk.endswith(("cx", "cy")) else np.float32))
    dd = np.load(f"{SRC}/{'banc_door.npz' if mode == 'banc' else 'male_door.npz'}")
    for odor in ODORS:
        w(f"{mode}_door_{odor}_idx", dd[odor + "_idx"].astype(np.int32))
        w(f"{mode}_door_{odor}_val", dd[odor + "_val"].astype(np.float32))
    man[mode] = {"N": int(d["N"][0]), "E": int(len(pre)), "sorted_by_post": True}
    del d, r, dd, pre, post, sw, o
    print(f"{mode} OK", flush=True)

json.dump(man, open(f"{DST}/manifest.json", "w"), indent=1)
print("OK manifest.json", flush=True)
