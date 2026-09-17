# Whole-Brain Drosophila as a Behaving Agent: Structure-Only Modeling with Measured Dynamics

**Status:** working sketch (repo: fly-brain, CC BY-NC 4.0). All numbers measured, deterministic, replicated Python/C++.

## 1. Premise

Frozen connectome (FlyWire FAFB v783 / BANC v888 / MCNS v1.0 / MANC v1.2.1),
frozen Dale signs (NT predictions), protocol-only dynamics (drive 2.0, top-k 5%,
thresholds, gains). No fitted weights. Question: how much behavior emerges from
structure + minimal protocol?

## 2. Learning is real synapses (third factor throughout)

- US pathway (`dan_rew/dan_pun` drives PAM/PPL, separation ~2.0 vs ~0.01).
- `train()` measures DAN and gates KC→MBON depression (Handler/Hige compartments):
  graded 0/+20.5/+35.1 for reward 0/0.5/1.0; `dan_block` gives exactly +0.0000
  (causal); reversal +21 → −7.4; identical to 6 decimals in C++.
- Sleep: SHY wash + synaptic tag (95% vs 59% retention) + replay consolidation
  (95% → 149%: sleep *enhances* memory, both langs to 3 decimals).
- R-STDP in spiking loop (Florian-like DA-EMA gate): no-US |dW| = 0.0 exactly,
  US ~1400/1540. No unsupervised Hebb remains anywhere in the codebase.
- Associability (Pearce-Hall novelty bonus, opt-in): 2x faster trial-1 on novel
  odors (+27 vs +13); REJECTED for markets (tune +, holdout collapses: novelty
  amplifies regime chop).

## 3. Ring attractor (central complex)

- EPG ring order recovered from connectivity topography (spectral seriation on
  PEN profiles; PEN→EPG spread 0.077 vs 0.73 shuffled). Old EB-xy wedge was
  wrong (0.746) and discarded.
- Rate: cue lands ±6, dark holds pinned, angvel integrates ±3 ranks/step.
- Spike: full triple in deterministic tonic regime (plateau + uniform tonic,
  no Poisson variance): dark pinned, av± exact. Key insight: Poisson background
  (2 Hz = pure noise at 0.4 spikes/window) seeded WTA jumps.
- Per-neuron homeostasis: compartment-local fan scaling SHIPPED; static size
  gain (kills MB) and intrinsic thresholds (too slow/suppress+jump) reverted
  with numbers. MCNS ring functional (validated=0).

## 4. Closed-loop trading (Drosophila as economic agent)

- 1 step = 1 day; 6 momentum features as odor; LONG/FLAT from MB; P&L via DAN;
  walk-forward, 2bp costs, deterministic.
- Base: learned helplessness (x0.97). Champion (live KC slots + H6 horizon +
  leak 0.3 + net-of-cost): SPY x2.78/+1.05/DD21% (BH 2.19/+0.68/52%),
  QQQ x3.29/+1.02/DD32% (BH 3.19/+0.82/55%).
- Tune 2018-21 +1.39; holdout 2022-24 +0.37 (BH x1.23). Cross-brain mb=MCNS
  identical (x1.12 both, 2024).
- Nulls (all measured): SHORT, more features (curse), colonies, sleep/replay/
  sizing/gates, hysteresis, stops, MPC (ret1 R2=0.006: EMH holds), vol sizing
  (overfit), novelty. Architecture ceiling ≈ Sharpe 1.0 (associative
  conditioning, not planning).

## 5. Open

Spike dark-hold needed plateau+tonic protocol (biological names, fitted scales);
MCNS ring unvalidated; FAFB synapse refresh unaudited; metacontroller unbuilt.
