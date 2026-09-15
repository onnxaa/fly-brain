"""Export frozen connectome data to flat little-endian binaries for fly_cpp.

C++ must not depend on Arrow/parquet/pandas: this script (run once, in
fly_gnn/) streams Connectivity_783.parquet in batches and writes raw
edge arrays + all role/group/odor tables + manifest.json into
../fly_cpp/data/.

Layout per array: <name>.<ext> where ext i32=Int32, f32=Float32,
i64=Int64, u8=uint8. manifest.json gives N/E/counts for each.
"""
import json
import os
import numpy as np

SRC = "."
DST = "../fly_cpp/data"
os.makedirs(DST, exist_ok=True)
man = {}


def w(name, arr):
    arr = np.ascontiguousarray(arr)
    ext = {"int32": "i32", "float32": "f32", "int64": "i64",
           "uint8": "u8"}[str(arr.dtype)]
    fn = f"{DST}/{name}.{ext}"
    arr.tofile(fn)
    man[name] = {"file": os.path.basename(fn), "n": int(arr.size),
                 "dtype": str(arr.dtype)}
    print(f"{name}: n={arr.size} dtype={arr.dtype}", flush=True)


# ---- mb mode ----
d = np.load(f"{SRC}/mb_circuit.npz")
w("mb_pre", d["pre"].astype(np.int32))
w("mb_post", d["post"].astype(np.int32))
w("mb_w", d["weight"].astype(np.float32))  # signed; C++ splits sign/|w|
for k in ("inputs_ALPN", "KC", "MBON", "DAN"):
    w(f"mb_{k}", d[k].astype(np.int32))
g = np.load(f"{SRC}/mb_groups_v3.npz")
for k in ("approach", "avoid", "dan_pam", "dan_ppl"):
    w(f"mb_{k}", g[k].astype(np.int32))
o = np.load(f"{SRC}/odor_glom.npz")
w("mb_odorA", o["odorA"].astype(np.int32))
w("mb_odorB", o["odorB"].astype(np.int32))
# fan-in scaling (same formula as enable_scaling mb branch)
fan = np.zeros(int(max(d["pre"].max(), d["post"].max()) + 1), np.float64)
np.add.at(fan, d["post"], np.abs(d["weight"].astype(float)))
fan[fan == 0] = 1.0
w("mb_fan", fan.astype(np.float32))
# KC->MBON analytic readout maps (same construction as FlyBrainAPI mb)
N = int(max(d["pre"].max(), d["post"].max()) + 1)
pre, post = d["pre"], d["post"]
KC = np.sort(d["KC"])
MBON = np.sort(d["MBON"])
kc_rank = {int(x): i for i, x in enumerate(KC)}
mb_rank = {int(x): i for i, x in enumerate(MBON)}
KCset = set(int(x) for x in d["KC"])
MBset = set(int(x) for x in MBON)
km_mask = np.array([(int(p) in KCset) and (int(q) in MBset)
                    for p, q in zip(pre, post)])
w("mb_km_ki", np.array([kc_rank[int(a)] for a in pre[km_mask]], np.int32))
w("mb_km_mi", np.array([mb_rank[int(b)] for b in post[km_mask]], np.int32))
avoid_set = set(int(x) for x in g["avoid"])
w("mb_km_is_avoid",
  np.isin(post[km_mask], list(avoid_set)).astype(np.uint8))
man["mb"] = {"N": N, "E": int(len(pre))}
del d, g, o, fan, km_mask

# ---- full mode roles/groups (mapped, same code as FlyBrainAPI full) ----
import pandas as pd
r = np.load(f"{SRC}/roles_full.npz")
comp = pd.read_csv(f"{SRC}/Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
mb = np.load(f"{SRC}/mb_circuit.npz")
mb_fly = mb["flywire_ids"]
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KCg = np.sort(loc2glob[mb["KC"]])
MBONg = np.sort(loc2glob[mb["MBON"]])
w("full_KC", KCg.astype(np.int32))
w("full_MBON", MBONg.astype(np.int32))
g2 = np.load(f"{SRC}/mb_groups_v3.npz")
w("full_approach",
  np.array([int(loc2glob[a]) for a in g2["approach"]], np.int32))
w("full_avoid",
  np.array([int(loc2glob[a]) for a in g2["avoid"]], np.int32))
w("full_dan_pam",
  np.array([int(loc2glob[a]) for a in g2["dan_pam"]
            if int(a) < len(loc2glob)], np.int32))
w("full_dan_ppl",
  np.array([int(loc2glob[a]) for a in g2["dan_ppl"]
            if int(a) < len(loc2glob)], np.int32))
for k in ("ORN", "MECH", "VIS", "ALPN", "EFFERENT", "DESC", "ME", "LO",
          "DESC_L", "DESC_R", "ORN_L", "ORN_R", "ALPN_L", "ALPN_R",
          "MECH_L", "MECH_R", "R", "VIS_eye"):
    if k in r:
        w(f"full_{k}", np.asarray(r[k], dtype=np.int32))
for k in ("R_cx", "R_cy"):
    w(f"full_{k}", np.asarray(r[k], dtype=np.float32))
Nf = int(r["N"][0])
man["full"] = {"N": Nf}
f = np.load(f"{SRC}/fan_abs.npz")
assert f["fan"].shape == (Nf,)
w("full_fan", f["fan"].astype(np.float32))
dd = np.load(f"{SRC}/door_odors.npz")
for odor in ("geosmin", "co2", "hexanone3", "methyl_salicylate",
             "butanedione", "ethyl_hexanoate"):
    w(f"door_{odor}_idx", dd[odor + "_idx"].astype(np.int32))
    w(f"door_{odor}_val", dd[odor + "_val"].astype(np.float32))
man["door_odors"] = ["geosmin", "co2", "hexanone3", "methyl_salicylate",
                     "butanedione", "ethyl_hexanoate"]
t = np.load(f"{SRC}/taste_grns.npz")
for k in ("sugar", "bitter", "ir94e", "water"):
    w(f"taste_{k}", t[k + "_idx"].astype(np.int32))

# ---- full edges: stream parquet -> flat (low RAM) ----
import pyarrow.parquet as pq
pf = pq.ParquetFile(f"{SRC}/Connectivity_783.parquet")
E = pf.metadata.num_rows
man["full"]["E"] = int(E)
print(f"full edges: E={E} N={Nf}", flush=True)
fpre = open(f"{DST}/full_pre.i32", "wb")
fpost = open(f"{DST}/full_post.i32", "wb")
fw = open(f"{DST}/full_w.f32", "wb")
for b in pf.iter_batches(batch_size=2000000,
                         columns=["Presynaptic_Index", "Postsynaptic_Index",
                                  "Excitatory x Connectivity"]):
    fpre.write(np.asarray(b["Presynaptic_Index"]).astype(np.int32).tobytes())
    fpost.write(np.asarray(b["Postsynaptic_Index"]).astype(np.int32).tobytes())
    fw.write(np.asarray(b["Excitatory x Connectivity"])
             .astype(np.float32).tobytes())
fpre.close()
fpost.close()
fw.close()
for n in ("full_pre", "full_post", "full_w"):
    ext = "f32" if n.endswith("_w") else "i32"
    man[n] = {"file": f"{n}.{ext}", "n": int(E),
              "dtype": "float32" if n.endswith("_w") else "int32"}
json.dump(man, open(f"{DST}/manifest.json", "w"), indent=1)
print("OK manifest.json", flush=True)
