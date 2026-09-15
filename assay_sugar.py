"""Asay Shiu na ICH kodzie (Brian2), nasze dane v783: sugar LB3 -> MN9? n_run=2."""
import pandas as pd, sys
sys.path.insert(0, ".")
from shiu_model import default_params, run_exp
params = dict(default_params)
params["n_run"] = 2
comp = pd.read_csv("Completeness_783.csv")
fly_ids = comp[comp.columns[0]].values.astype("int64")
flyset = set(int(f) for f in fly_ids)
sup = pd.read_csv("supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
sugar = [int(r) for r in sup[sup["cell_type"] == "LB3"]["root_id"] if int(r) in flyset]
print(f"sugar LB3: {len(sugar)}", flush=True)
run_exp("sugar_lb3", sugar, "./results", "Completeness_783.csv",
        "Connectivity_783.parquet", params=params, n_proc=2, force_overwrite=True)
print("DONE-assay", flush=True)
