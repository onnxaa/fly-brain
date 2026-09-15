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

## Parity (measured)

- Rate path bit-parity mb+full: step/train/lif/tmaze/spaced identical to
  Python to 4 decimals (tmaze 23s vs ~5min Python; full step ~0.7s vs ~20s).
- X-zone growth: same behavior class, different bits (PCG64 vs mt19937_64)
  and hardened fan for new sinks (sum|w|, Python crashes when scaling ran
  before growth — C++ stays correct).
- Habituate/flee absolute scale differs by training RNG (drop 60% both,
  recovery both); heatbox/buridan/detour match behavior class.

RNG note: Python uses PCG64, C++ mt19937_64 — rates agree within sampling
noise, not bit-wise (X-zone growth + gust/shuffle trajectories only).
Rate path (relu/lif) is bit-parity verified mb+full.
