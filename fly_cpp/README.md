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

RNG note: Python uses PCG64, C++ mt19937_64 — rates agree within sampling
noise, not bit-wise (X-zone growth only). Rate path (relu/lif) is bit-parity
verified mb+full.
