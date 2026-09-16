"""Sequence order (mb): A->B vs B->A via leaky working memory.

set_state(leak) carries activity across steps (default 0 = classic
stateless, bit-identical). X readout taught on the 2nd step discriminates
presentation order — the architecture's first temporal memory.
"""
from fly_api import FlyBrainAPI


def main():
    b = FlyBrainAPI(mode="mb", path=".")
    assert abs(b.step(odor="A")["MB_pref"] - (-4.0642)) < 1e-3, "stateless regression"
    print("stateless regression ok (-4.0642)", flush=True)
    b.set_state(0.3)
    b.x_add_output("seq", n=16, seed=5, per_in=64)
    for _ in range(6):
        b.reset_state(); b.step(odor="A")
        b.x_teach_output("seq", target=1.0, trials=1, eta=0.7, odor="B")
        b.reset_state(); b.step(odor="B")
        b.x_teach_output("seq", target=0.0, trials=1, eta=0.7, odor="A")
    outs = []
    for seq in (("A", "B"), ("B", "A")):
        b.reset_state(); b.step(odor=seq[0])
        outs.append(b.step(odor=seq[1]).get("X_seq", 0))
    print(f"A->B: {outs[0]:.3f} B->A: {outs[1]:.3f} sep={outs[0]-outs[1]:+.3f}", flush=True)
    print("PASS-seq" if outs[0] > outs[1] + 0.2 else "FAIL-seq", flush=True)


if __name__ == "__main__":
    main()
