# fly_cpp — C++ 1:1 twin of fly_api.py

Superfast whole-brain Drosophila (frozen topology/Dale, protocol-only learning).
No Arrow/parquet here — data comes as flat little-endian binaries.

## Export data (once, in repo root)

```bash
python3 export_flat.py     # mb/full edges + roles + fan + DoOR + taste -> ../fly_cpp/data/
python3 export_spike.py    # APL/graded sets + mb APL extension -> ../fly_cpp/data/
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

Weights: C++ uses raw float32 `.wbin` (save/load_wbin), not Python `.npz`.
`buridan`/`habituate`/`detour` train inline when `--w-in` is missing.
`heatbox` writes the punishment schedule for `heatbox_ctl` (yoked/unpaired).

## Fast protocol tests (mb, seconds)

```bash
./build/fly --data <DATA> test-x      # X-zone discrimination, drift ~0
./build/fly --data <DATA> test-std    # STD off flat, on ~60% habituate, recover
./build/fly --data <DATA> test-sleep  # SHY wash toward baseline
```

## Parity (measured, 8 CPU, Sept 2026)

- Rate path bit-parity mb+full: step/train/lif/tmaze/spaced identical to
  Python to 4 decimals.
- full load: 11.2s (Python, pandas+parquet) → ~3.0s (flat binaries) ≈ 3–4x.
- full step in-process: Python 0.5s (numpy scatter is already near-optimal)
  vs C++ ~0.7–1.0s — parity, no compute win claimed.
- full one-shot (cold load + 1 step): ~12–15s → ~3.8s ≈ 3–4x.
- mb one-shot: ~1.0s → 0.39s ≈ 2.6x; mb step in-process: parity (~0.02s).
- tmaze end-to-end: 23s in-process (24 forwards, CSR rebuilds on train).
- Memory full: Python 1505 MB → C++ ~525 MB ≈ 3x (no pandas/pyarrow,
  no per-forward temporaries).
- Real wins: load time, memory, one-shot latency, single static binary
  with no data deps. Per-step FLOPs are at parity — headroom (mmap,
  quantization, GPU) not yet exploited.
- X-zone growth: same behavior class, different bits (PCG64 vs mt19937_64)
  and hardened fan for new sinks (sum|w|, Python crashes when scaling ran
  before growth — C++ stays correct).
- Habituate/flee absolute scale differs by training RNG (drop 60% both,
  recovery both); heatbox/buridan/detour match behavior class.

RNG note: Python uses PCG64, C++ mt19937_64 — rates agree within sampling
noise, not bit-wise (X-zone growth + gust/shuffle trajectories only).
Rate path (relu/lif) is bit-parity verified mb+full.
