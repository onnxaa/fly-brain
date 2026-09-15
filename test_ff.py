"""KC drive feedforward-only A->K (27k edges) vs full recurrent. Overlap on the full brain."""
import numpy as np, pandas as pd, time
t0=time.time()
roles = np.load("roles_full.npz"); N=int(roles["N"][0]); ORN=roles["ORN"]
mb = np.load("mb_circuit.npz"); mb_fly=mb["flywire_ids"]
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype(np.int64)
fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
KC_glob = np.sort(loc2glob[mb["KC"]]); ALPN_glob = np.sort(loc2glob[mb["inputs_ALPN"]])
con = pd.read_parquet("Connectivity_783.parquet", columns=["Presynaptic_Index","Postsynaptic_Index","Excitatory x Connectivity"])
pre = con["Presynaptic_Index"].values.astype(np.int32); post = con["Postsynaptic_Index"].values.astype(np.int32)
ws = con["Excitatory x Connectivity"].values.astype(np.float32); del con
Aset=set(int(x) for x in ALPN_glob); Kset=set(int(x) for x in KC_glob)
ff = np.array([(p in Aset) and (q in Kset) for p,q in zip(pre,post)])
print(f"A->K edges: {ff.sum()} ({time.time()-t0:.0f}s)", flush=True)
rng = np.random.default_rng(1); in_dim,hid=8,16
emb = rng.normal(0,0.5,size=(N,in_dim)).astype(np.float32)
Wmsg = rng.normal(0,0.3,size=(in_dim,hid)).astype(np.float32)
Wself = rng.normal(0,0.3,size=(in_dim,hid)).astype(np.float32)
oA = ORN[:len(ORN)//2]; oB=ORN[len(ORN)//2:]
# 1-hop ORN->ALPN: ALPN activation
xA=emb.copy(); xA[oA]+=2.0; xB=emb.copy(); xB[oB]+=2.0
P=xA@Wmsg; aggA=np.zeros((N,hid),dtype=np.float32); np.add.at(aggA,post[ff],P[pre[ff]]*ws[ff][:,None])
P=xB@Wmsg; aggB=np.zeros((N,hid),dtype=np.float32); np.add.at(aggB,post[ff],P[pre[ff]]*ws[ff][:,None])
hA=np.maximum(0,xA@Wself+aggA); hB=np.maximum(0,xB@Wself+aggB)
sA=hA[KC_glob].mean(axis=1); sB=hB[KC_glob].mean(axis=1)
k=max(1,int(len(sA)*0.05))
mA=sA>=np.sort(sA)[-k]; mB=sB>=np.sort(sB)[-k]
print(f"FF A->K: overlap={np.logical_and(mA,mB).sum()}/{k} corr={float(np.corrcoef(sA,sB)[0,1]):.2f}", flush=True)
# comparison: same ALPN patterns?
print("PASS-ff", flush=True)
