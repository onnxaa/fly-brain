"""Kompartmenty: MBON approach = zdominowane przez KC unerwiane PAM; avoid = przez KC unerwiane PPL1.
Motyw D->K (47k) x K->M (62k)."""
import numpy as np, pandas as pd, time
t0=time.time()
mb = np.load("mb_circuit.npz"); mb_fly=mb["flywire_ids"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
g = np.load("mb_groups.npz")
pam_loc = set(int(x) for x in g["dan_pam"]); ppl_loc = set(int(x) for x in g["dan_ppl"])
pam_glob = set(int(loc2glob[l]) for l in pam_loc); ppl_glob = set(int(loc2glob[l]) for l in ppl_loc)
KC_glob = set(int(x) for x in np.sort(loc2glob[mb["KC"]])); MB_glob = set(int(x) for x in np.sort(loc2glob[mb["MBON"]]))
con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values; post = con["Postsynaptic_Index"].values; w = np.abs(con["Excitatory x Connectivity"].values.astype(float)); del con
print(f"load ({time.time()-t0:.0f}s)", flush=True)
# D->K: waga PAM/PPL per KC
pam_in = {}; ppl_in = {}
for a,b,c in zip(pre,post,w):
    if int(b) in KC_glob:
        if int(a) in pam_glob: pam_in[int(b)] = pam_in.get(int(b),0)+c
        elif int(a) in ppl_glob: ppl_in[int(b)] = ppl_in.get(int(b),0)+c
print(f"D->K: KC z PAM={len(pam_in)} z PPL={len(ppl_in)}", flush=True)
# per MBON: suma przez KC
score = {}
for a,b,c in zip(pre,post,w):
    if int(a) in KC_glob and int(b) in MB_glob:
        s = pam_in.get(int(a),0) - ppl_in.get(int(a),0)
        score[int(b)] = score.get(int(b),0) + s*c
app = sorted([m for m,v in score.items() if v >= 0]); avo = sorted([m for m,v in score.items() if v < 0])
print(f"approach(PAM-dom)={len(app)} avoid(PPL-dom)={len(avo)}", flush=True)
old_app = set(int(loc2glob[a]) for a in g["approach"])
print(f"zgodnosc ze starym parzyste/nieparzyste: {len(set(app)&old_app)}/{len(app)} (ok. 50% = heurystyka byla losowa)", flush=True)
# map do local-MB (format mb_groups)
glob2loc = {int(gg): int(l) for l,gg in enumerate(loc2glob)}
np.savez("mb_groups_v2.npz",
         approach=np.array(sorted(glob2loc[m] for m in app),dtype=np.int32),
         avoid=np.array(sorted(glob2loc[m] for m in avo),dtype=np.int32),
         dan_pam=g["dan_pam"], dan_ppl=g["dan_ppl"])
print("saved mb_groups_v2.npz", flush=True)
vals = sorted(score.values())
print(f"rozklad score: min={vals[0]:.0f} med={vals[len(vals)//2]:.0f} max={vals[-1]:.0f}", flush=True)
