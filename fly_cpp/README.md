# fly_cpp — superfast C++ 1:1 twin of the Fly Brain 🪰

Whole-brain *Drosophila* (138k neurons / 15M synapses, frozen FlyWire
topology/Dale, protocol-only learning) — 16× faster load, 8× faster train,
~3× less RAM vs Python. No Arrow/parquet here — data comes as flat little-endian binaries.

## Export data (once, in repo root)

```bash
python3 export_flat.py     # mb/full edges + roles + fan + DoOR + taste -> ../fly_cpp/data/
python3 export_spike.py    # APL/graded sets + mb APL extension -> ../fly_cpp/data/
python3 export_vnc.py      # real VNC (MANC) + DESC type-bridge -> ../fly_cpp/data/
python3 export_cns.py      # BANC (female) + MCNS (male) whole-CNS -> ../fly_cpp/data/
```

`fly_cpp/data/*.i32/*.f32` are git-ignored (182M). Run the two scripts above
to regenerate. `manifest.json` documents N/E/counts.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Toolchain: g++ 15.2, cmake 3.31, OpenMP. RAM ~1.5G for full forward.

## Use (phase1: info/step/train)

```bash
./build/fly --data ../fly_cpp/data --mode mb info
./build/fly --data ../fly_cpp/data --mode mb step --odor A
./build/fly --data ../fly_cpp/data --mode full step --odor geosmin
./build/fly --data ./data --mode mb train --odor A --reward 1.0
```

## Use (phase 2: batteries, full mode)

```bash
./build/fly --data <DATA> tmaze [--w-out tmaze.wbin]
./build/fly --data <DATA> train_fix [--w-out etho.wbin]
./build/fly --data <DATA> buridan [--w-in etho.wbin] [--steps 25] [--n-switch 3]
./build/fly --data <DATA> habituate [--w-in arena.wbin]
./build/fly --data <DATA> detour [--w-in etho.wbin] [--steps 20]
./build/fly --data <DATA> heatbox [--sched hb_sched.bin] [--steps 100] [--test 40]
./build/fly --data <DATA> heatbox_ctl [--sched hb_sched.bin] [--steps 100] [--test 40]
./build/fly --data <DATA> spaced
```

```bash
./build/fly --data <DATA> --mode full --vnc --mech-bin mech.f32 step
# mech.f32 = float32 vector over full MECH pool (export order); prints
# VNC_desc/VNC_motor/legL/legR/imb/wing/neck (bit-parity vs Python enable_vnc)
```

## Whole-CNS modes (banc/mcns, bit-parity vs Python)

```bash
./build/fly --data <DATA> --mode banc --odor geosmin step
./build/fly --data <DATA> --mode mcns --odor geosmin step
./build/fly --data <DATA> --mode mcns bench --steps 3
```

Prints `BANC_motor/legL/legR/imb/wing/neck` (intact brain→VNC chain, one
animal). Measured (8 CPU): banc N=150k/E=11M load 0.9s step ~60ms train
~130ms; mcns N=165k/E=25.6M load 2.4s step ~166ms train ~384ms.
Parity: banc geosmin MB +0.0532, mcns −4.3808, motor pools — all identical
to Python. `--vpol 1/2` (ON/OFF channels, bg 0.5): bit-exact too
(ON bright 0.65201/0.62182, OFF dark 0.02949/0.08326). Batteries (tmaze/...) stay full-mode (FAFB-validated); bench
works in every mode.

## Scaling regimes (`--scaling static|active`)

`static` (default): divide by total |fan-in| — every number in this repo.
`active`: divide by ACTIVE |fan-in| only — signal survives any depth.
Use active for deep vision (`--hops 4`): banc bars reach KC/DN/legs
(DN~0.10, legs lateralized, wing~0.21). No free parameters, no avalanche
(peak 4 hops, graceful decay to 8). Default stays static (all validations).
Parity: bit-identical at 2 hops (banc +10.1793, mcns −365.8993); at 4+
hops behavior-class (means match ~10%, MB_pref exact value diverges —
difference of two large means + FP order chaos; lateralization preserved).
Full mode too (no code change needed): active bars reach DN with correct
ipsilateral lateralization (left 0.0587/0.0581, right 0.0521/0.0634).
Rejected: tonic baseline (tested 0.5 in all modes — adds literally nothing;
uniform tonic dies by E/I cancellation within ~3 hops; OFF responses would
need chain-wide tonicity, out of scope).

## MCNS retinotopy via BANC homology + wbin headers

MCNS flat files carry no R coordinates — but BANC `malecns_match` links 384
MCNS R neurons to BANC R7/R8 (100% type-consistent, side agreement 54-0
where known). Their RF transfers by homology (rank-norm among themselves);
the rest stay eye-mean. Result: mcns train_fix ±1.44 monotonic (was ±1.0
saturated eye-split), buridan STEER 65% vs BASE 33%. Propagation past R
still dies in rate mode (histamine synapse, same as FAFB) — fix readouts
tap R directly. Lesson logged: an earlier run drove hemifields while
reading retinotopic pools (mush) — drive and readout must match.

wbin v2: 40B header (magic FLYWBIN1 + mode + N + E); cross-mode loads fail
fast (`wbin mode mismatch: file=mcns brain=banc`); legacy headerless files
still load (size check as before).

## Central complex (compass + steering readouts)

CX pools from data (BANC EPG50/Delta7-40/PEN42/PFN443; MCNS 50/42/42/456).
EPG ring order = spectral (PEN-profile topography, no coordinates):
BANC PEN->EPG spread 0.077 + EPG-EPG locality 0.31 (validated=1);
MCNS 0.273/0.14 (functional, validated=0, EXPERIMENTAL). PFL side clean in
BANC, by DESC-output wiring in MCNS (29/21). NT: EPG/PEN ACh (+), D7
glutamate (-) - Dale signs already in weights.
Ring attractor (`test-attractor`, rate): cx_cue landmark -> bump lands
(+/-6 ranks), dark holds (banc pinned, mcns drifts <=10), angvel walks it
with correct sign (lumpy individual, ~1-2 ranks/step). Mechanism: rotation
integrator (direction from PEN data: left-PEN +1.5 ranks) + K=2 maintenance
over frozen EPG+D7+PEN weights (fan-norm, raw EM ratios, CX-local carry
0.85). Spike: cue-evoked bump only (recipe spk2.0/GI2/u.08/bg2/plat.8);
spike dark-hold/velocity = OPEN. Per-neuron homeostasis status: SHIPPED =
compartment-local fan equalization (EB neurons scale EB inputs; flattens
20x PEN wells); TRIED+REVERTED = static size gain (kills MB propagation,
LIF cliff) and intrinsic-plasticity thresholds (too slow at ETA<=0.1,
suppress+jump at 0.5) - see fly_api notes. VOL per-neuron data (Google
segmentation, build_vol.py) stored for future compartment models.
Legacy CX_bump mapping bug fixed (pool indexed by rank -> argsort); legacy
assays unaffected (loop armed only by cx_cue/angvel/set_cx_gain).

## Batteries across modes (CNS-agnostic protocols)

`tmaze|spaced|heatbox(_ctl)|test-x|test-std|test-sleep` take `--mode` since
this update (visual batteries stay full-only: no R front-end in banc/mcns;
clock-gating stays full-only: no clock pools yet). Measured (8 CPU):

| battery | banc (F) | mcns (M) |
|---|---|---|
| tmaze | learn ✓ PI→1.00, reversal ✗ (stays +1.00) | all ✓ incl. reversal −0.23 |
| test-x | PASS 2.34/0.88 | PASS 1.86/0.21 |
| test-std | PASS 61% drop | PASS 61% drop |
| test-sleep | PASS (SHY wash) | PASS (SHY wash) |
| heatbox | FAIL 80=80 (MB scale ~10x too small for −2.0 threshold) | PASS 22 vs 80 |
| spaced | runs, PI ceiling +1.00 | massed 0.26 ≈ spaced 0.25, retention flat |

Heatbox places: ALPN prosthesis in full, fixed odor-mixture prosthesis
(seed 11) in banc/mcns — MECH subsets don't reach MB in rate mode.

## Vision in banc/mcns (real R front-ends)

BANC: 1827 R7/R8 with RF from meta `position` (rank-norm, same recipe as
FAFB) + L1/L2 luminance proxy at column RF (R1-6 absent in v888 — labeled
proxy). MCNS: 4107 R1-6/R7/R8 as honest eye split only (rootSide L/R; no
coordinates in v1.0 flat files, no fake RF). All visual batteries take
`--mode`: train_fix (banc monotonic ±1.27, mcns saturated ±1.0), buridan
(banc STEER 68% vs BASE 33%, mcns 56% vs 33%), habituate (banc drop 59%,
mcns 27%, both recover), detour (final err 8°/6°). Deep rate vision still
attenuates (OL→CB ~2000x) — fix/flee readouts tap R directly, same trick
as full mode (whose rate KC/DN from vision are also ~0).

Weights: C++ uses raw float32 `.wbin` (save/load_wbin), not Python `.npz`.
`buridan`/`habituate`/`detour` train inline when `--w-in` is missing.
`heatbox` writes the punishment schedule for `heatbox_ctl` (yoked/unpaired).

## Spike, clock, taste in banc/mcns

- Spike: own APL + graded early-vision sets per dataset (no more FAFB graft),
  exported as `<mode>_spike_apl/gset`, loaded per mode in C++ too. mcns SFA
  calibration mirrored (ainc=16): geosmin MBON ~4 Hz, KC 15% (was 521 Hz/100%).
  Spike uses Poisson drive — behavior-class parity by design, not bit-exact.
  Plasticity works in spike too (same rules on measured Hz, plus real
  trace-STDP with --stdp: A+/A-=0.005/0.0052, tau 20ms, burn-excluded,
  true spikes only, clip [0.05,650]; unsupervised Hebbian, valence still
  in DAN slab). mb curves: plain 9.11->9.51->9.51 (saturates) vs STDP
  8.97->9.45->9.72 (keeps climbing); mcns 1 trial +0.14->+0.74.
  Python STDP validated too (mb 9.05->10.50->7.90, mcns +0.34/7.5min);
  C++ is the practical path (OpenMP, seconds).
  sleep/clock are mode-agnostic; train() validated mb (9.05->9.67->7.83)
  and C++ mcns (+0.14->+0.49, fast — Python whole-brain spike-train is
  minutes per trial, use C++). True per-synapse STDP: open work.
  banc healthy out of the box (odor ~73Hz max); mcns defaults SFA ainc=16
  (calibrated MBON<70Hz, KC 16-49% — denser than Shiu MB-only, open work).
- Clock: TTFL pools per mode (s/l-LNv morning, LNd/DN1 evening); spaced
  batteries run fully clock-gated in every CNS mode.
- Taste: single GRN pool per mode (BANC 1423, MCNS 1428) — no quality split,
  both datasets lack Gr receptor annotation. All four names drive it
  (labeled EXPERIMENTAL); sugar bit-parity Python=C++ (DAN_pam 0.0029).

## Fast protocol tests (mb, seconds)

```bash
./build/fly --data <DATA> test-x      # X-zone discrimination, drift ~0
./build/fly --data <DATA> test-std    # STD off flat, on ~60% habituate, recover
./build/fly --data <DATA> test-sleep  # SHY wash toward baseline
```

## Perf (measured, 8 CPU, warm cache; Python = numpy rate path)

| op (full brain) | Python | C++ | gain |
|---|---|---|---|
| load | 11.2s | ~0.7s | **16x** |
| step | 0.50s | ~0.11s | **4–5x** |
| train (punish) | 1.27s | 0.16s | **8x** |
| sleep(5) | 1.40s | 0.20s | **7x** |
| tmaze end-to-end | — | 5.5s (was 23s) | — |
| spaced end-to-end | — | 62s (was >120s) | — |
| RAM | 1505 MB | ~537 MB | **~3x** |

What did it (all parity-safe, no `-ffast-math`, no FP reassociation):
- Flat binaries + CSR-direct layout: edges stably sorted by `post` once in
  `export_flat.py` (counting sort, O(E)). Stable = within-post order kept,
  so per-post FP summation sequences are bit-identical to Python.
  Edge index IS the CSR slot: no scatter, no permutation map, sequential
  passes everywhere. Unsorted input still works via general fallback.
- `train`/`sleep`/wash write through to CSR in place (`csr_update_edge`,
  fused loops) instead of rebuilding + reallocating the whole 15M CSR.
- `build_km` bitmap instead of 30M `binary_search`.
- Persistent forward buffers (no 3×N alloc per step), activation enum
  (no per-element `strcmp`), readout indices precomputed once.
- LTO (`CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE`).
- Headroom left: unify `pre`/`csr_pre` (−60 MB), mmap data, GPU — not done.
- X-zone growth: same behavior class, different bits (PCG64 vs mt19937_64)
  and hardened fan for new sinks (sum|w|, Python crashes when scaling ran
  before growth — C++ stays correct).
- Habituate/flee absolute scale differs by training RNG (drop 60% both,
  recovery both); heatbox/buridan/detour match behavior class.

RNG note: Python uses PCG64, C++ mt19937_64 — rates agree within sampling
noise, not bit-wise (X-zone growth + gust/shuffle trajectories only).
Rate path (relu/lif) is bit-parity verified mb+full.
