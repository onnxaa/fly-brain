"""Per-neuron volume (Google BANC segmentation) into roles.

IN: banc_meta.feather (volume_nm3) + banc_roles.npz.
OUT: banc_roles.npz += VOL (K,) float32, local-idx aligned, NaN kept
     (API imputes median at load). MCNS annot has no volume -> male VOL
     stays absent (API falls back to gain 1.0, documented).

Biophysics: input resistance ~ 1/size -> small neurons need less current.
Used as static excitability prior (gain = (median/vol)^0.5) in spike mode.
"""

import numpy as np
import pandas as pd

print("meta...", flush=True)
m = pd.read_feather("banc_meta.feather")
pr = m["proofread"].astype(str).isin(["TRUE", "True", "true", "1"]).values
glia = (np.asarray(m["super_class"].fillna("").astype(str), dtype=str) == "glia")
mk = m[pr & (~glia)].reset_index(drop=True)
K = len(mk)
vol = pd.to_numeric(mk["volume_nm3"], errors="coerce").values.astype(np.float32)
print(f"kept {K}, NaN frac={np.isnan(vol).mean():.3f}, median={np.nanmedian(vol):.3g}", flush=True)
r = np.load("banc_roles.npz", allow_pickle=True)
d = {k: r[k] for k in r.files}
assert int(d["K"][0]) == K, (d["K"][0], K)
d["VOL"] = vol
np.savez_compressed("banc_roles.npz", **d)
print("banc_roles.npz + VOL", flush=True)
