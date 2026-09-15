# fly_cpp — superfast C++ 1:1 twin of the Fly Brain 🪰

Whole-brain *Drosophila* (138k neurons / 15M synapses, frozen FlyWire
topology/Dale, protocol-only learning) — 16× faster load, 8× faster train,
~3× less RAM vs Python. No Arrow/parquet here — data comes as flat little-endian binaries.

## Export data (once, in repo root)

```bash
python3 export_flat.py     # mb/full edges + roles + fan + DoOR + taste -> ../fly_cpp/data/
python3 export_spike.py    # APL/graded sets + mb APL extension -> ../fly_cpp/data/
python3 export_vnc.py      # real VNC (MANC) + DESC type-bridge -> ../fly_cpp/data/
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

Weights: C++ uses raw float32 `.wbin` (save/load_wbin), not Python `.npz`.
`buridan`/`habituate`/`detour` train inline when `--w-in` is missing.
`heatbox` writes the punishment schedule for `heatbox_ctl` (yoked/unpaired).

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
