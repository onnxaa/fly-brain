"""Prawdziwe profile DoOR: macierz konsensusowa -> glomerulus -> ORN globals. Ujemne (inhibicja) zachowane."""
import pandas as pd, numpy as np
M = pd.read_csv("door_matrix.csv", sep=";", index_col=0)
print(f"macierz: {M.shape}", flush=True)
mp = pd.read_csv("door_mappings.csv", sep=";")
unit2glom = {}
for _, r in mp.iterrows():
    g = str(r["glomerulus"])
    if g in ("?", "", "nan") or pd.isna(r["glomerulus"]):
        continue
    for col in ["receptor", "code", "code.OSN", "OSN"]:
        u = str(r[col])
        if u not in ("?", "", "nan") and not pd.isna(r[col]):
            unit2glom.setdefault(u, g)
# kolumny macierzy -> glomerulus
col2g = {c: unit2glom.get(c) for c in M.columns}
print(f"kolumn: {len(M.columns)}, zmapowanych: {sum(v is not None for v in col2g.values())}", flush=True)
od = pd.read_csv("door_odor.csv", sep=";")
CAS = {"geosmin": "19700-21-1", "co2": "124-38-9", "hexanone3": "589-38-8",
       "methyl_salicylate": "119-36-8", "butanedione": "431-03-8", "ethyl_hexanoate": "123-66-0"}
for k, v in CAS.items():
    hit = od[od["CAS"].astype(str) == v]
    print(k, "->", hit["Name"].values if len(hit) else "BRAK", flush=True)
