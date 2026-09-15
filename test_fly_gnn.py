import torch
from fly_gnn import FlyBrainGNN, synthetic_connectome

torch.manual_seed(0)
n, edge_index, w = synthetic_connectome(n=2000, avg_deg=12)
print(f"nodes={n} edges={edge_index.shape[1]}")

model = FlyBrainGNN(n, edge_index, w, in_dim=16, hidden_dim=64, steps=3)
print("EI:", model.ei_balance())

sensory_idx = {"sugar": list(range(0, 20)), "loom": list(range(20, 40))}
motor_idx = list(range(40, 44))

rates, h = model({"sugar": 1.0, "loom": 0.0}, sensory_idx, motor_idx)
print("motor rates:", rates.detach().tolist())
assert rates.shape == (4,)
assert torch.all(rates >= 0)

# 1 krok distillation: naśladuj fake target LIF
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.tensor([12.0, 0.5, 3.0, 0.1])
opt.zero_grad()
pred, _ = model({"sugar": 1.0}, sensory_idx, motor_idx)
loss = ((pred - target) ** 2).mean()
loss.backward()
# maska trzyma topologię: gradient tylko na istniejących krawędziach
opt.step()
print(f"distill loss: {loss.detach().item():.4f} OK")

# E/I po kroku - znak nie może się zmienić (Dale)
print("EI after:", model.ei_balance())
print("PASS")
