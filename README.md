# Fly Brain — a data-only whole-brain model of *Drosophila*

A working simulation of 138,639 neurons / 15,091,983 synapses from the FlyWire
FAFB v783 connectome (Dorkenwald et al., Nature 2024), with LIF dynamics after
Shiu et al., Nature 2024.

**Core rule (`pure` mode, the default): 100% of topology, Dale signs and |w|
come from the data.** Only protocol constants are free (stimulus drive, KC
top-k 5%, DAN anti-Hebb 0.85, thresholds). Anything that breaks the rule lives
in the explicitly opt-in **X zone** and is logged in a deviation ledger.

> **Data license: CC BY-NC 4.0 — non-commercial use only.**

## What this is

- `fly_api.py` — the whole thing behind one class, `FlyBrainAPI(mode="mb"|"full")`.
  `mb` = mushroom-body circuit (6,289 neurons, ~50 steps/s, interactive).
  `full` = entire brain (~1 s/step, ~1 GB RAM, needs the parquet).
- A behavior readout stack (odor preference, tropotaxis steering, motor/lateral
  readouts), a plasticity stack (reward/punishment, protective DAN, SHY sleep,
  autonomous sleep, circadian TTFL), and a validation battery (~20 assay scripts
  with `PASS-` verdicts, numbers in `results/`).

## How it works

**Forward.** Pure diffusion over the real graph, no learned parameters:
`a = ReLU(Σw·a_pre / fan_in − thr) + drive`, 1–2 hops for mushroom-body
readouts, up to 4 for deep (descending/motor) readouts. Fan-in scaling
(`fan_abs.npz`) keeps total drive bounded — this is why shallow paths are stable
and very deep chains attenuate (documented limit, worked around with spiking).

**Smell (real).** 6 odors from the DoOR 2.0 consensus matrix (691×78, inhibition
kept), mapped ORN→glomerulus via `door_mappings.csv`, broadcast to 2,279 ORNs
(`door_real.py` → `door_odors.npz`). Geosmin CAS corrected to 16423-19-1.

**Learning.** `train(reward/punish)` = whole-graph R-Hebb (±2% + renorm) plus
DAN anti-Hebb LTD on KC→MBON (Huang 2024: PAM/slab-avoid for reward, PPL1 for
punishment). `gated=True` (default) scales depression by KC uniqueness against
registered codes (`x_remember`) — shared memory protected (conflict retention
117%), with zero codes it is bit-identical to the legacy slab.

**Memory.** `sleep()` = SHY homeostatic downscaling toward baseline; `forgetting
curve +2.61→+0.83`, identity drift 0.03%. `enable_auto_sleep()` = Borbély
two-process (pressure from KC activity + light zeitgeber, hysteresis bouts,
drive gating ×0.2). `enable_clock()` + `tick_clock(hours)` = artificial Goldbeter
TTFL (free-run 23h, LD entrainment); only biochemistry outside the connectome,
fed into the correct M (s-LNv/l-LNv) vs E (LNd/DN1) neurons. **Clock time flows
only through `tick_clock`** — behavior steps never advance it.

**Vision.** `image=` drives R1-6/R7/R8 photoreceptors (10,582 cells) — each R
samples the input at its own receptive field from real pos_x/pos_y
(rank-normalized, bilinear + Gaussian RF, any HxW, no pixel grid); downstream
ME/LO/LC/DN run through the real graph. No lens/ommatidial optics model.
Motion (Reichardt) is readout-only, never
injected. Spiking seeing in `see_lif.py`: graded early vision (38,871
non-spiking cells, tonic release) + full-brain LIF — flash drives LC 0→21Hz
and DNs; loom recruits DNp01 2x over recede.

**Activation.** Two rate modes: `relu` (default, fast legacy `max(0,x)`) and
`lif` (`set_activation("lif")`, saturating LIF-shaped
`SAT*(1-exp(-x/SAT))`, Shiu `t_mbr/t_rfc` behind the shape; exact log-form
rejected — measured to crush contrast on fan-scaled drives). Weak-drive
numbers match within ~2% (all validations transfer); strong drives saturate
(5x overload: MB 723→198); learning direction preserved. Spike timing stays
in `see_lif.py`/Brian.

**Spike mode (3rd).** `set_activation("spike", Tms=200, seed=7, wdrv=68.75,
ainc=4, rmax=150, adapt=0.2, burn=50)` — full LIF sim 1:1 with `test_lif.py`
v2 (g-reset, 2-step refractory/delay, signed weights, spiking APL x4, graded
early vision + tonic release, sensory adaptation, SFA, onset burn-in); same
dict in Hz. mb works (halves: ALPN 65Hz, KC ~15%, MB d=+3.5Hz). Full runs
(~1min/200ms) with an honest envelope: strong sustained drives ignite
recurrent loops (geosmin: ALPN ~100Hz, KC ~77%, MBON ~130Hz saturate), weak
drives die — a bistable cliff, not a bug (APL cache verified, inhibition
genuinely overwhelmed: ORN→ALPN feedforward too weak at physiological rates
while ALPN-loop feedback too strong once firing). Knobs tame it partially
(SFA: MBON 137→15Hz). Full-brain spike gain control (STD/LN co-tuning) is
open work — no reference data exists (Shiu validated MB only).

**Spikes.** `test_lif.py` = NumPy port of the Brian/Shiu equations (g-reset,
refractory, delay, signed weights, spiking APL): KC 5.1% vs Brian 6.5%.
`loom_lif.py` runs the same dynamics on the full 138k brain (~100 s/500 ms):
sugar→MN9 60Hz, loom→DNp01/DNp04, conflict suppression 100%.

**Growth (X zone).** `x_grow_assoc` (any modality pool → any target),
`x_neurogenesis` (interference-gated), `x_prune_kc`, `x_add/cut_edge`,
`x_report()` ledger. 10k-neuron tranche protocol keeps 97% of learning
performance; touch modality goes from unlearnable (d=0) to learned.

## Purpose

A testable platform for: associative learning curves, reversal, hard/easy
discrimination, extinction, generalization vs DoOR similarity, mixture
tradeoffs, odor-vs-threat conflict (LC4/LPLC2 drive → escape synergy + MN9
suppression), lateral tropotaxis (`turn_olf`), sleep/forgetting, circadian
gating, and continual growth — all against a frozen-data identity check
(topology, Dale 100%, weight drift <5%).

## Use

```python
from fly_api import FlyBrainAPI
mb = FlyBrainAPI(mode="mb")
mb.step(odor="A")["MB_pref"]; mb.train(odor="A", reward=1.0)

fly = FlyBrainAPI(mode="full"); fly.enable_scaling()
o = fly.step(odor="ethyl_hexanoate", odor_left="geosmin", image=img, mech=v)
fly.train(odor="ethyl_hexanoate", reward=1.0)
fly.save_weights("w.npz")
```

Inputs combine freely: `odor` (DoOR name / 'A'/'B' / vector), `odor_left/right`,
`image` (full only), `mech`, `mech_left/right`, `alpn`, `dan_rew/dan_pun`.
Outputs: `MB_pref/app/avo`, `MBON(96)`, `KC_active`, modality means,
`ALPN_L/R`, `ORN_L/R`, `MECH_L/R`, `turn_olf`, `DN_L/R`, `turn`, `motor_pref`,
`EFFERENT(1481)`, `EI_sum`, sleep/clock state.

## Validation (measured)

Topology N/E exact; Dale 100%; drift 0.12%, corr 0.99. DoOR→ALPN sim-vs-1hop
corr 0.60–0.94. Acquisition d −5.65→+7.89; reversal crosses zero in 1–2 trials;
30-trial rewire holds at +20.86 with 0.06% drift. Lesions, retina symmetry,
tropotaxis 0.20→0.90 all PASS. Full numbers: `results/*.txt`,
checkpoints: `weights_registry.json`.

## Limits

No lens/ommatidial optics (point sample + Gaussian RF only); motion readout-only, not injected; no body/VNC (flight only via prosthesis); DN-turn at
noise floor (steer via `MB_pref`/`turn_olf`); full spiking is NumPy-only
(~100 s/500 ms, Brian OOMs); rate mode loses deep chains; no forgetting without
sleep; molecular clock/TTFL outside data; DNa02 steering needs LAL drive.
