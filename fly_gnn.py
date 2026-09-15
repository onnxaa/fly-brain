"""Fly brain -> GNN (bez PyG, sam torch sparse).
Działa na CPU. Pełne 138k/15M też wejdzie w CSR (~120MB), ale trenuj na subcircuit.
Format wejściowy jak Connectivity_783.parquet:
  Presynaptic_Index, Postsynaptic_Index, Excitatory x Connectivity
"""
import torch
import torch.nn as nn


class FlyGNNLayer(nn.Module):
    def __init__(self, n_in, edge_index, edge_weight, edge_sign,
                 hidden_dim=64, aggr="sum"):
        super().__init__()
        self.n_in = n_in
        self.aggr = aggr
        # topologia zamrożona jako buffer
        self.register_buffer("edge_index", edge_index.clone())  # [2, E] pre->post
        self.register_buffer("edge_sign", edge_sign.clone().float())  # +/-1
        self.register_buffer("edge_mag", edge_weight.clone().float().abs())
        # trenowalne tylko magnitudy (Dale: znak stały)
        self.log_mag = nn.Parameter(torch.log(edge_weight.float().abs() + 1e-6))
        self.lin_self = nn.Linear(n_in, hidden_dim)
        self.lin_msg = nn.Linear(n_in, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [N, n_in]
        pre, post = self.edge_index[0], self.edge_index[1]
        mag = torch.exp(self.log_mag) * self.edge_sign  # zachowuje znak
        # message: x_pre * w
        msg_src = self.lin_msg(x[pre]) * mag.unsqueeze(1)
        agg = torch.zeros(x.size(0), msg_src.size(1), device=x.device)
        agg.index_add_(0, post, msg_src)
        out = self.lin_self(x) + agg
        return torch.nn.functional.leaky_relu(self.lin_out(torch.relu(out)), 0.1)


class FlyBrainGNN(nn.Module):
    def __init__(self, num_nodes, edge_index, edge_weight,
                 in_dim=16, hidden_dim=64, steps=3,
                 n_sensory_types=4, n_motor=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.steps = steps
        self.sign = torch.sign(edge_weight.float())
        self.sign[self.sign == 0] = 1.0
        self.node_emb = nn.Embedding(num_nodes, in_dim)
        # typ neuronu jako prosty one-hot projekcja (sensory/motor/inter)
        self.layers = nn.ModuleList([
            FlyGNNLayer(
                in_dim if s == 0 else hidden_dim,
                edge_index, edge_weight.abs(), self.sign,
                hidden_dim=hidden_dim,
            )
            for s in range(steps)
        ])
        self.motor_head = nn.Linear(hidden_dim, 1)

    def forward(self, sensory_dict, sensory_idx, motor_idx):
        # sensory_dict: {name: float 0..1} -> wstrzykujemy jako bias do emb
        x = self.node_emb.weight.clone()
        for name, val in sensory_dict.items():
            idx = sensory_idx.get(name, [])
            if len(idx) and val != 0:
                x[idx] += float(val) * 2.0
        h = x
        for layer in self.layers:
            h = layer(h)
        motor_rates = self.motor_head(h[motor_idx]).squeeze(1)
        return torch.nn.functional.softplus(motor_rates), h

    def ei_balance(self):
        with torch.no_grad():
            mag = torch.exp(self.layers[0].log_mag) * self.layers[0].edge_sign
            return {
                "sum": float(mag.sum()),
                "mean": float(mag.mean()),
                "pos": float((mag > 0).sum()),
                "neg": float((mag < 0).sum()),
            }

    # ---- online R-Hebb (bez backpropa, w locie) ----
    def init_online(self, eta=0.01, alpha=1e-4, decay=0.9,
                    w_min=0.05, w_max=5.0):
        layer = self.layers[0]
        e = layer.edge_index.shape[1]
        self._ol = {
            "eta": eta, "alpha": alpha, "decay": decay,
            "w_min": w_min, "w_max": w_max,
            "elig": torch.zeros(e),
        }
        with torch.no_grad():
            mag = torch.exp(layer.log_mag).detach()
            post = layer.edge_index[1]
            ref = torch.zeros(self.num_nodes)
            ref.index_add_(0, post, mag)
            self._ol["ref_in"] = ref  # referencyjna suma |w| per neuron

    @torch.no_grad()
    def online_step(self, sensory_dict, sensory_idx, motor_idx, reward=0.0):
        rates, h = self.forward(sensory_dict, sensory_idx, motor_idx)
        ol = getattr(self, "_ol", None)
        if ol is None:
            return rates, h
        layer = self.layers[0]
        # koincydencja na AKTYWACJACH WARSTWY 0 (nie ostatniej):
        x0 = self.node_emb.weight.clone()
        for name, val in sensory_dict.items():
            idx = sensory_idx.get(name, [])
            if len(idx) and val != 0:
                x0[idx] += float(val) * 2.0
        h1 = layer(x0)
        a0 = x0.mean(dim=1)
        a1 = h1.mean(dim=1)
        pre, post = layer.edge_index[0], layer.edge_index[1]
        coinc = a0[pre] * a1[post]
        ol["elig"].mul_(ol["decay"]).add_(coinc)
        if reward != 0.0:
            dlog = ol["eta"] * float(reward) * ol["elig"] - ol["alpha"]
            layer.log_mag.add_(dlog)
            layer.log_mag.clamp_(
                float(torch.log(torch.tensor(ol["w_min"]))),
                float(torch.log(torch.tensor(ol["w_max"]))),
            )
            # renormalizacja energii per neuron: suma |w| wraca do ref
            mag = torch.exp(layer.log_mag)
            cur = torch.zeros(self.num_nodes)
            cur.index_add_(0, post, mag)
            scale = torch.ones(self.num_nodes)
            nz = cur > 1e-9
            scale[nz] = (ol["ref_in"][nz] / cur[nz]).clamp(0.8, 1.25)
            layer.log_mag.add_(torch.log(scale[post]))
        return rates, h


def load_flywire_parquet(path_con, max_nodes=None):
    """Ładuje prawdziwy parquet do edge_index/weight. Wymaga pandas+pyarrow."""
    import pandas as pd
    con = pd.read_parquet(path_con)
    pre = torch.tensor(con["Presynaptic_Index"].values, dtype=torch.long)
    post = torch.tensor(con["Postsynaptic_Index"].values, dtype=torch.long)
    w = torch.tensor(con["Excitatory x Connectivity"].values, dtype=torch.float32)
    n = int(max(int(con["Presynaptic_Index"].max()), int(con["Postsynaptic_Index"].max())) + 1)
    if max_nodes is not None:
        keep = (pre < max_nodes) & (post < max_nodes)
        pre, post, w = pre[keep], post[keep], w[keep]
        n = max_nodes
    edge_index = torch.stack([pre, post], dim=0)
    return n, edge_index, w


def synthetic_connectome(n=2000, avg_deg=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    e = n * avg_deg
    pre = torch.randint(0, n, (e,), generator=g)
    post = torch.randint(0, n, (e,), generator=g)
    # Balans per sieć: połowa +k, połowa -k o tej samej magnitudzie
    # (w prawdziwym konektomie pilnuj tego per neuron docelowy, nie globalnie)
    mag = torch.randint(1, 6, (e,), generator=g).float()
    sign = torch.ones(e)
    sign[1::2] = -1.0  # parzyste -, nieparzyste +
    w = sign * mag
    edge_index = torch.stack([pre, post], dim=0)
    return n, edge_index, w
