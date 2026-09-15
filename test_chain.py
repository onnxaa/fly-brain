"""FF chain: h1 = 1-hop(ORN bias) -> KC drive = A->K from h1[ALPN]. Overlap + corr per layer."""
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
rng = np.random.default_rng(1); in_dim,hid=8,16
emb = rng.normal(0,0.5,size=(N,in_dim)).astype(np.float32)
Wmsg = rng.normal(0,0.3,size=(in_dim,hid)).astype(np.float32)
Wself = rng.normal(0,0.3,size=(in_dim,hid)).astype(np.float32)
Wmsg2 = rng.normal(0,0.3,size=(hid,hid)).astype(np.float32)
# GLOMERULAR ODORS: whole ORN types (like real odorants), not random neuron halves
ao = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id","cell_class","cell_type"])
ao = ao[ao["cell_class"]=="olfactory"]
fly2idx2 = {int(f): i for i, f in enumerate(pd.read_csv("Completeness_783.csv").iloc[:,0].values.astype(np.int64))}
typy = sorted(t for t in ao["cell_type"].unique() if isinstance(t, str))
have = set(fly2idx2.keys())
oA = np.array(sorted({fly2idx2[int(r)] for _, r in ao[ao["cell_type"].isin(typy[:len(typy)//2])]["root_id"].items() if int(r) in have}), dtype=np.int32)
oB = np.array(sorted({fly2idx2[int(r)] for _, r in ao[ao["cell_type"].isin(typy[len(typy)//2:])]["root_id"].items() if int(r) in have}), dtype=np.int32)
oA = oA[np.isin(oA, ORN)]; oB = oB[np.isin(oB, ORN)]
print(f"glomeruli A={len(typy[:len(typy)//2])} types/{len(oA)} ORN  B={len(typy[len(typy)//2:])} types/{len(oB)} ORN", flush=True)
np.savez("odor_glom.npz", odorA=oA, odorB=oB)
CH=2000000
def h1_of(bias):
    x=emb.copy(); x[bias]+=2.0
    P=x@Wmsg; agg=np.zeros((N,hid),dtype=np.float32)
    for s in range(0,len(pre),CH):
        e=slice(s,min(s+CH,len(pre)))
        np.add.at(agg,post[e],P[pre[e]]*ws[e][:,None])
    return np.maximum(0,x@Wself+agg)
hA=h1_of(oA); hB=h1_of(oB)
aA=hA[ALPN_glob].mean(axis=1); aB=hB[ALPN_glob].mean(axis=1)
print(f"ALPN corr A/B={float(np.corrcoef(aA,aB)[0,1]):.2f} ({time.time()-t0:.0f}s)", flush=True)
PA=hA@Wmsg2; aggA=np.zeros((N,hid),dtype=np.float32); np.add.at(aggA,post[ff],PA[pre[ff]]*ws[ff][:,None])
PB=hB@Wmsg2; aggB=np.zeros((N,hid),dtype=np.float32); np.add.at(aggB,post[ff],PB[pre[ff]]*ws[ff][:,None])
xA=emb.copy(); xA[oA]+=2.0; xB=emb.copy(); xB[oB]+=2.0
kA=np.maximum(0,xA@Wself+aggA)[KC_glob].mean(axis=1)
kB=np.maximum(0,xB@Wself+aggB)[KC_glob].mean(axis=1)
k=max(1,int(len(kA)*0.05))
ov=np.logical_and(kA>=np.sort(kA)[-k],kB>=np.sort(kB)[-k]).sum()
print(f"KC-via-FF-chain: overlap={ov}/{k} corr={float(np.corrcoef(kA,kB)[0,1]):.2f}", flush=True)
print("PASS-chain", flush=True)
