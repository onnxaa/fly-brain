# Fly Brain — data-only model całego mózgu *Drosophila*

Symulacja 138 639 neuronów / 15 091 983 synaps z konektomu FlyWire FAFB v783
(Dorkenwald i in., Nature 2024). Zasada: **100% topologii, znaków Dale'a i |w|
z danych** w trybie `pure` — strojeniu podlegają tylko stałe protokołu
(drive, top-k 5%, anti-Hebb 0.85, progi). Równania LIF za Shiu i in., Nature 2024.

> Licencja danych: **CC BY-NC 4.0 — tylko użytek niekomercyjny.**
> Moby w grach: tak, ale niekomercyjnie (prototypy, edukacja).

## Zależności

Python 3, `numpy`, `pandas`, `pyarrow` (parquet), opcjonalnie `brian2`
(torcje kolcowe MB). Start w katalogu z danymi.

## Quickstart

```python
from fly_api import FlyBrainAPI

# 1) Szybki obwód MB (6k neuronów, ~50 kroków/s) — zapachy A/B, nauka
mb = FlyBrainAPI(mode="mb")
print(mb.step(odor="A")["MB_pref"])
mb.train(odor="A", reward=1.0)   # R-Hebb + DAN anti-Hebb na K->M

# 2) Cały mózg (~1 s/krok, ~1 GB RAM) — prawdziwe zapachy DoOR, obraz, dotyk, smak
fly = FlyBrainAPI(mode="full")
fly.enable_scaling()             # cache fan-in (raz)
o = fly.step(odor="ethyl_hexanoate", image=img64, mech=mvec)
# o: MB_pref, MB_app/avo, MBON(96), KC_active, ALPN/ORN/VIS/MECH_mean,
#    DN_L/R, turn, motor_pref, EFFERENT(1481), ALPN_L/R, turn_olf, ...
fly.train(odor="ethyl_hexanoate", reward=1.0)
fly.save_weights("moje.npz"); fly.load_weights("moje.npz")
```

## Wejścia / wyjścia

Wejścia `step()/train()` (dowolna kombinacja): `odor` (nazwa DoOR / 'A'/'B' /
wektor), `odor_left/right` (boczne, full), `image` (64×64, full),
`mech`, `mech_left/right`, `alpn` (drive wprost), `dan_rew/dan_pun`.
Wyjścia: preferencja `MB_pref`, składowe approach/avoid, wektory MBON/DAN/EFFERENT,
średnie modalności, `turn_olf = ALPN_L−ALPN_R`, `KC_active`, `EI_sum`.

## Pamięć, sen, zegar

- `train(reward/punish)` — R-Hebb po grafie + DAN anti-Hebb (LTD K→M, Huang 2024);
  `gated=True` (default) chroni zapamiętane kody (`x_remember`).
- `sleep(episodes)` — homeostaza SHY (dryf do baseline); `enable_auto_sleep()` —
  sen autonomiczny (Borbély: presja z KC + bramka światła), log w `_sleep_log`.
- `enable_clock()` + `tick_clock(hours)` — sztuczny TTFL (Goldbeter): free-run 23h,
  entrainment LD. **Czas dobowy płynie tylko przez `tick_clock`** (kroki to triale).

## Eksperymenty (skrypty)

`test_lesions.py`, `test_loop.py` (tropotaksja), `test_retina.py`, `test_lif.py`
(LIF 1:1 Brian, PASS: KC 5.1%), `loom_exp.py` + `loom_lif.py` (ucieczka vs karmienie:
supresja MN9 100%), `see_lif.py` (widzenie graded+LIF), `learn_battery.py` (8 testów
pamięci), `rewire_long.py` (30 triali), `test_grow.py` (strefa X).
Wyniki liczbowe: `results/*.txt`. Wagi: `weights_registry.json`.

## Strefa X (poza pure, jawny opt-in)

`x_grow_assoc` (wzrost dla dowolnej modalności), `x_neurogenesis` (bramkowana
interferencją), `x_prune_kc`, `x_add/cut_edge`, `x_report()` (księga odchyleń),
`x_code/x_novelty/x_remember`. Default API nigdy ich nie tyka.

## Walidacja (wybrane)

| test | wynik |
|---|---|
| topologia / Dale / dryf | N/E, 100%, 0.12%, corr 0.99 |
| DoOR.real → ALPN | corr sim-vs-1hop 0.60–0.94, KC overlap 0.8–16% |
| LIF MB | KC 0.38Hz/5.1%, MBON ~2 (Brian: 0.39/6.5%/2.2) |
| nauka full | d −5.65→+7.91; reversal w 1–2 triale; konflikt 117% |
| looming | DNp01/DNp04 synergia; MN9 60Hz→0 (100%) |
| widzenie | R 65Hz→LC 21Hz, DN 48 vs 24 (loom/recede) |
| sen/sen-auto/zegar | krzywa zapominania; cykle ~24h; LD 2 doby |

## Ograniczenia (uczciwe)

Wzrok-wejście to placeholder (retinotopia z danych, optyka nie); brak ciała/VNC
(lot tylko protezą); DN-turn w podłodze szumu (sterowanie przez `MB_pref`/`turn_olf`);
kolcowy full-brain tylko numpy (~100 s/500 ms, Brian OOM); rate gubi głębokie
łańcuchy (fan-dzielenie); model nie zapomina bez snu; TTFL/ER2-zegar poza danymi.
