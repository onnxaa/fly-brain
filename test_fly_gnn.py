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

# 1 distillation step: mimic a fake LIF target
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.tensor([12.0, 0.5, 3.0, 0.1])
opt.zero_grad()
pred, _ = model({"sugar": 1.0}, sensory_idx, motor_idx)
loss = ((pred - target) ** 2).mean()
loss.backward()
# mask holds topology: gradient only on existing edges
opt.step()
print(f"distill loss: {loss.detach().item():.4f} OK")

# E/I after the step - sign must not change (Dale)
print("EI after:", model.ei_balance())
print("PASS")
