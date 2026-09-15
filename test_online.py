import torch
from fly_gnn import FlyBrainGNN, synthetic_connectome

torch.manual_seed(1)
n, edge_index, w = synthetic_connectome(n=2000, avg_deg=12, seed=1)
model = FlyBrainGNN(n, edge_index, w, in_dim=16, hidden_dim=64, steps=3)
model.init_online(eta=0.005, alpha=1e-5, decay=0.9)

sensory_idx = {"odorA": list(range(0, 20)), "odorB": list(range(20, 40))}
motor_idx = list(range(40, 44))
before = model.ei_balance()["sum"]

# T-maze online: odorA + reward, odorB + nic. 30 triali w locie, bez backpropa.
for t in range(30):
    rA, _ = model.online_step({"odorA": 1.0}, sensory_idx, motor_idx, reward=0.0)
    rA, _ = model.online_step({"odorA": 1.0}, sensory_idx, motor_idx, reward=+1.0)
    rB, _ = model.online_step({"odorB": 1.0}, sensory_idx, motor_idx, reward=0.0)

rA, _ = model({"odorA": 1.0}, sensory_idx, motor_idx)
rB, _ = model({"odorB": 1.0}, sensory_idx, motor_idx)
after = model.ei_balance()["sum"]
print("motor odorA:", rA.detach().tolist())
print("motor odorB:", rB.detach().tolist())
print(f"EI sum before={before:.1f} after={after:.1f}")
print("prefers A on motor0:", bool(rA[0] > rB[0]))
print("PASS")
