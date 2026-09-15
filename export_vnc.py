"""Export real VNC (MANC v1.2.1) to flat little-endian binaries for fly_cpp.

Reads vnc_circuit.npz + vnc_bridge.npz (see build_vnc.py) and writes
CSR-direct layout (stable sort by post => bit-identical FP order with
Python np.add.at) + pools + bridge + manifest entries into ../fly_cpp/data/.
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


d = np.load(f"{SRC}/vnc_circuit.npz")
M = int(d["N"][0])
pre = np.asarray(d["pre"]).astype(np.int32)
post = np.asarray(d["post"]).astype(np.int32)
sw = (np.asarray(d["weight"]).astype(np.float32)
      * np.asarray(d["sign"]).astype(np.float32))
o = np.argsort(post, kind="stable")
w("vnc_pre", pre[o])
w("vnc_post", post[o])
w("vnc_w", sw[o])  # signed; C++ splits sign/|w| like full_w
w("vnc_fan", np.asarray(d["fan"]).astype(np.float32))
for k in ("desc", "motor_all", "leg_L", "leg_R", "wing_L", "wing_R", "neck"):
    w(f"vnc_{k}", np.asarray(d[k]).astype(np.int32))
b = np.load(f"{SRC}/vnc_bridge.npz")
w("vnc_bridge_b", np.asarray(b["brain"]).astype(np.int32))
w("vnc_bridge_v", np.asarray(b["vnc"]).astype(np.int32))
man["vnc"] = {"M": M, "E": int(len(pre)), "shared_types": int(b["n_shared"][0]),
              "bridge_pairs": int(len(b["vnc"])), "sorted_by_post": True}
json.dump(man, open(f"{DST}/manifest.json", "w"), indent=1)
print(f"vnc M={M} E={len(pre)} OK manifest.json", flush=True)
