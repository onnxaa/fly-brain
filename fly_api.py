"""Pelne API mozgu muchy (numpy, bez torch).

DANE: FlyWire FAFB v783 (Dorkenwald et al. Nature 2024, CC BY-NC 4.0 - TYLKO UZYTEK NIEKOMERCYJNY),
  adnotacje flyconnectome (supp1.tsv), profile DoOR 2.0 (Munch & Galizia 2016),
  listy GRN Shiu et al. Nature 2024, walencje MBON Aso et al. eLife 2014.
ZASADA: zero parametrow spoza danych - topologia, znaki Dale'a i |w| z konektomu;
  jedyne stale to protokol eksperymentu (drive, top-k 5%, anti-Hebb, thr).
Tryb pure (domyslny) = czysta dyfuzja; legacy rate (pure=False) zachowany do porownan.

Wejscia sensoryczne (dowolna kombinacja naraz):
  image   : macierz HxW float 0..1 (skala szarosci) -> neurony VIS (placeholder retinotopii:
            block-average do 64x64=4096 pikseli na pierwsze 4096 neuronow VIS, reszta VIS cicha)
   odor    : wektor (n_ORN,) float 0..1 LUB slowo 'A'/'B' (polowki ORN, jak w testach)
   odor_left/odor_right : lateralny zapach DoOR (nazwa) LUB wektor na strone
             (full + ORN_L/R z laterality.py; laczy sie addytywnie z odor)
             Wyjscia boczne: ALPN_L/ALPN_R, ORN_L/R, turn_olf = ALPN_L-ALPN_R
   mech_left/mech_right : lateralny dotyk, wektory na strone (full + MECH_L/R);
             wyjscia MECH_L/R
  mech    : wektor (n_MECH,) float 0..1 (dotyk/wibracje)
  alpn    : bezposredni drive glomerularny (n_ALPN,) - omija ORN (1-hop do KC w trybie mb)
  dan_rew : float 0..1 (PAM, nagroda), dan_pun : float 0..1 (PPL1, kara)

Wyjscia (dict z step()/train()):
  MB_pref, MB_app, MB_avo, MBON (96,), DAN_pam, DAN_ppl, KC_active, KC_overlap_info,
  motor_pref, motor_A, motor_B, EFFERENT (1481,), ALPN_mean, VIS_mean, MECH_mean, ORN_mean, EI_sum

Tryby:
  mode='mb'   : obwod MB 6289/548k, 1-hop, ~1-2 s/krok (domyslny, interaktywny)
  mode='full' : caly mozg 138639/15M, 2-hop + gain control, ~20 s/krok (parquet stream)

Nauka: train() robi R-Hebb (full: caly graf +-2% + renorm; mb: tylko K->M)
  + DAN anti-Hebb na K->M (nagroda slab-avoid, kara slab-approach).
Sen: sleep() robi homeostatyczne skalowanie w dol do baseline (SHY) - zapominanie.
Topologia i znaki Dale'a zamrozone - tozsamosc mozgu zachowana (patrz validate_flyness.py).
"""
import numpy as np

IMG_N = 64 * 64

class FlyBrainAPI:
    def __init__(self, mode="mb", seed=1, path="."):
        assert mode in ("mb", "full")
        self.mode, self.path = mode, path
        rng = np.random.default_rng(seed)
        self.in_dim, self.hid = 8, 16
        if mode == "mb":
            d = np.load(f"{path}/mb_circuit.npz")
            self.pre, self.post = d["pre"], d["post"]
            w = d["weight"].astype(np.float32)
            self.N = int(max(self.pre.max(), self.post.max()) + 1)
            self.ALPN, self.KC, self.MBON, self.DAN = d["inputs_ALPN"], d["KC"], d["MBON"], d["DAN"]
            import os as _os
            _gf = f"{path}/mb_groups_v3.npz" if _os.path.exists(f"{path}/mb_groups_v3.npz") else f"{path}/mb_groups.npz"
            g = np.load(_gf)
            self.approach, self.avoid = g["approach"], g["avoid"]
            self.dan_pam, self.dan_ppl = g["dan_pam"], g["dan_ppl"]
            # w trybie mb brak VIS/MECH/ORN/EFFERENT -> mapujemy sensory na ALPN:
            # odor->ALPN (gl direct), image->losowy podzbior ALPN? Nie: obraz w mb nie ma sensu,
            # wiec rzutujemy obraz na KC przez losowa projekcje (placeholder optyki w MB).
            self.VIS = np.array([], dtype=np.int32); self.MECH = np.array([], dtype=np.int32)
            self.ORN = np.array([], dtype=np.int32); self.EFFERENT = np.array([], dtype=np.int32)
            self._img_proj = rng.normal(0, 1.0, size=(IMG_N, min(256, len(self.KC)))).astype(np.float32)
            self._img_kc = np.sort(self.KC)[:self._img_proj.shape[1]]
        else:
            r = np.load(f"{path}/roles_full.npz")
            self.N = int(r["N"][0])
            self.ORN, self.MECH, self.VIS = r["ORN"], r["MECH"], r["VIS"]
            self.ALPN, self.EFFERENT = r["ALPN"], r["EFFERENT"]
            # lateralizacja wechowa (laterality.py, side z supp1.tsv; brak = puste)
            self.ORN_L = r["ORN_L"] if "ORN_L" in r else np.zeros(0, np.int32)
            self.ORN_R = r["ORN_R"] if "ORN_R" in r else np.zeros(0, np.int32)
            self.ALPN_L = r["ALPN_L"] if "ALPN_L" in r else np.zeros(0, np.int32)
            self.ALPN_R = r["ALPN_R"] if "ALPN_R" in r else np.zeros(0, np.int32)
            self.MECH_L = r["MECH_L"] if "MECH_L" in r else np.zeros(0, np.int32)
            self.MECH_R = r["MECH_R"] if "MECH_R" in r else np.zeros(0, np.int32)
            self.DESC_L = r["DESC_L"] if "DESC_L" in r else np.zeros(0, np.int32)
            self.DESC_R = r["DESC_R"] if "DESC_R" in r else np.zeros(0, np.int32)
            self.VIS_pix = r["VIS_pix"] if "VIS_pix" in r else np.tile(np.arange(IMG_N), (len(self.VIS)+IMG_N-1)//len(self.VIS))[:len(self.VIS)]
            self.VIS_eye = r["VIS_eye"] if "VIS_eye" in r else np.zeros(len(self.VIS), np.int32)
            self.ME = r["ME"] if "ME" in r else np.zeros(0, np.int32)
            self.LO = r["LO"] if "LO" in r else np.zeros(0, np.int32)
            self._last_small = None  # do detektora ruchu (T4/T5-like)
            mb = np.load(f"{path}/mb_circuit.npz"); mb_fly = mb["flywire_ids"]
            import pandas as pd
            comp = pd.read_csv(f"{path}/Completeness_783.csv")
            fly_ids = comp[comp.columns[0]].values.astype(np.int64)
            fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
            loc2glob = np.array([fly2idx[int(f)] for f in mb_fly], dtype=np.int32)
            self.KC = np.sort(loc2glob[mb["KC"]]); self.MBON = np.sort(loc2glob[mb["MBON"]])
            import os as _os2
            _gf2 = f"{path}/mb_groups_v3.npz" if _os2.path.exists(f"{path}/mb_groups_v3.npz") else f"{path}/mb_groups.npz"
            g = np.load(_gf2)
            self.approach = np.array([int(loc2glob[a]) for a in g["approach"]])
            self.avoid = np.array([int(loc2glob[a]) for a in g["avoid"]])
            self.dan_pam = np.array([int(loc2glob[a]) for a in g["dan_pam"] if int(a) < len(loc2glob)], dtype=np.int32)
            self.dan_ppl = np.array([int(loc2glob[a]) for a in g["dan_ppl"] if int(a) < len(loc2glob)], dtype=np.int32)
            self._img_proj = None
            import pyarrow.parquet as pq
            self._pf = pq.ParquetFile(f"{path}/Connectivity_783.parquet")
        # wspolne: wagi K->M + embeddingi
        if mode == "mb":
            self.sign = np.sign(w); self.sign[self.sign == 0] = 1.0
            self.wM = np.abs(w)
            isK = np.zeros(self.N, bool); isK[self.KC] = True
            isM = np.zeros(self.N, bool); isM[self.MBON] = True
            self.km_mask = isK[self.pre] & isM[self.post]
            mb_s = np.sort(self.MBON); self._mb_pos = {m: i for i, m in enumerate(mb_s)}
            kc_s = np.sort(self.KC); self._kc_pos = {kk: i for i, kk in enumerate(kc_s)}
            self.km_ki = np.array([self._kc_pos[a] for a in self.pre[self.km_mask]], dtype=np.int32)
            self.km_mi = np.array([self._mb_pos[b] for b in self.post[self.km_mask]], dtype=np.int32)
        else:
            import pandas as pd
            con = pd.read_parquet(f"{path}/Connectivity_783.parquet",
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
            self.pre = con["Presynaptic_Index"].values.astype(np.int32)
            self.post = con["Postsynaptic_Index"].values.astype(np.int32)
            ws = con["Excitatory x Connectivity"].values.astype(np.float32); del con
            self.sign = np.sign(ws); self.sign[self.sign == 0] = 1.0
            self.wM = np.abs(ws); del ws
            self.E = len(self.pre)
            KCset = set(int(x) for x in self.KC); MBset = set(int(x) for x in self.MBON)
            self.km_mask = np.array([(p in KCset) and (q in MBset) for p, q in zip(self.pre, self.post)])
            kc_rank = {int(gg): i for i, gg in enumerate(self.KC)}
            mb_rank = {int(gg): i for i, gg in enumerate(self.MBON)}
            kp = self.pre[self.km_mask]; qp = self.post[self.km_mask]
            self.km_ki = np.array([kc_rank[int(a)] for a in kp], dtype=np.int32)
            self.km_mi = np.array([mb_rank[int(b)] for b in qp], dtype=np.int32)
            self._mb_pos = mb_rank; self._kc_pos = kc_rank
        avoid_set = set(int(x) for x in self.avoid)
        qp_all = self.post[self.km_mask]  # raz (bylo: kopia maski w kazdej z 62k iteracji!)
        self.km_is_avoid = np.isin(qp_all, list(avoid_set))
        self.km_is_approach = ~self.km_is_avoid
        nMB = len(self.MBON)
        self.ref_mb = np.zeros(nMB); np.add.at(self.ref_mb, self.km_mi, self.wM[self.km_mask].astype(float))
        self.emb = rng.normal(0, 0.5, size=(self.N, self.in_dim)).astype(np.float32)
        self.Wmsg = rng.normal(0, 0.3, size=(self.in_dim, self.hid)).astype(np.float32)
        self.Wself = rng.normal(0, 0.3, size=(self.in_dim, self.hid)).astype(np.float32)
        if mode == "full":
            self.Wmsg2 = rng.normal(0, 0.3, size=(self.hid, self.hid)).astype(np.float32)
            self.Wself2 = rng.normal(0, 0.3, size=(self.hid, self.hid)).astype(np.float32)
            self.Wmsg3 = rng.normal(0, 0.3, size=(self.hid, self.hid)).astype(np.float32)
            self.Wself3 = rng.normal(0, 0.3, size=(self.hid, self.hid)).astype(np.float32)
            self.ref_in = np.zeros(self.N); np.add.at(self.ref_in, self.post, self.wM.astype(float))
            self.elig = np.zeros(self.E, dtype=np.float32)
        self.wM0 = self.wM.copy()  # baseline snu (SHY): tylko dewiacje uczenia dryfuja
        self.CH = 2000000
        # sen autonomiczny (model dwuprocesowy Borbelyego): S=hormostat z aktywnosci mozgu,
        # C=bramka swiatla (zeitgeber). Skalar S jak elig (slad pozaneuronowy), decyzja budzi/sen
        # wynika ze stanu mozgu + srodowiska, nie od eksperymentatora. Domyslnie wylaczony.
        self._auto_sleep = None
        self.sleep_S = 0.0
        self.asleep = False
        self._ambient = 0.5
        self._sleep_log = []

    # ---- sensory ----
    def _img_to_vec(self, image):
        a = np.asarray(image, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        h, w = a.shape
        bh, bw = max(1, h//64), max(1, w//64)
        small = a[:64*bh, :64*bw].reshape(64, bh, 64, bw).mean(axis=(1, 3))
        return np.clip(small.reshape(-1), 0, 1)

    def _motion_energies(self, small):
        """Reichardt (T4/T5-like): 4 kierunki z pary klatek 16x16. Zwraca dict R/L/U/D >=0."""
        s = small.reshape(16, 16) if small.size == 256 else small.reshape(16, 16)
        out = {}
        if self._last_small is None:
            self._last_small = s.copy()
            return {"R": 0.0, "L": 0.0, "U": 0.0, "D": 0.0}, True
        p = self._last_small
        out["R"] = float(np.maximum(0, s*np.roll(p, 1, axis=1) - s*np.roll(p, -1, axis=1)).sum())
        out["L"] = float(np.maximum(0, s*np.roll(p, -1, axis=1) - s*np.roll(p, 1, axis=1)).sum())
        out["D"] = float(np.maximum(0, s*np.roll(p, 1, axis=0) - s*np.roll(p, -1, axis=0)).sum())
        out["U"] = float(np.maximum(0, s*np.roll(p, -1, axis=0) - s*np.roll(p, 1, axis=0)).sum())
        self._last_small = s.copy()
        return out, False

    def _door_lateral(self, odor, side):
        """Wycinek profilu DoOR na strone (L/R): maska idx do ORN_L/R. Tylko full."""
        import os as _os4
        _door = f"{self.path}/door_odors.npz"
        if not _os4.path.exists(_door) or odor not in (
                "geosmin", "co2", "hexanone3", "methyl_salicylate", "butanedione", "ethyl_hexanoate"):
            raise ValueError(f"odor boczny '{odor}': tylko zapachy DoOR (door_odors.npz)")
        _dd = np.load(_door)
        oi, ov = _dd[odor + "_idx"], (_dd[odor + "_val"] * 2.0).astype(np.float32)
        keep = np.isin(oi, side)
        return oi[keep], ov[keep]

    def encode(self, image=None, odor=None, mech=None, alpn=None, odor_left=None, odor_right=None,
               mech_left=None, mech_right=None):
        """Zwraca (idx, val, info): info ma 'motion' gdy policzono ruch."""
        info = {}
        idx, val = [], []
        if image is not None:
            self._ambient = float(np.asarray(image, dtype=np.float32).mean())
        if odor is not None:
            if isinstance(odor, str):
                import os as _os3
                _of = f"{self.path}/odor_glom.npz"
                _door = f"{self.path}/door_odors.npz"
                _taste = f"{self.path}/taste_grns.npz"
                if _os3.path.exists(_taste) and odor in ("sugar", "bitter", "water", "ir94e"):
                    if self.mode != "full":
                        raise ValueError("smaki GRN wymagaja mode='full'")
                    _tt = np.load(_taste)
                    sel = _tt[odor+"_idx"]
                    idx.append(sel); val.append(np.full(len(sel), 2.0, np.float32))
                elif _os3.path.exists(_door) and odor in (
                        "geosmin", "co2", "hexanone3", "methyl_salicylate", "butanedione", "ethyl_hexanoate"):
                    if self.mode != "full":
                        raise ValueError(f"zapach '{odor}' (DoOR, ORN) wymaga mode='full'; w mb uzyj 'A'/'B' lub wektora")
                    _dd = np.load(_door)
                    idx.append(_dd[odor+"_idx"]); val.append((_dd[odor+"_val"]*2.0).astype(np.float32))
                else:
                    if _os3.path.exists(_of) and len(self.ORN) and odor in ("A", "B"):
                        _od = np.load(_of)
                        sel = _od["odorA"] if odor == "A" else _od["odorB"]
                    else:
                        o = self.ORN if len(self.ORN) else self.ALPN
                        half = len(o)//2
                        sel = o[:half] if odor == "A" else o[half:]
                    idx.append(sel); val.append(np.full(len(sel), 2.0, np.float32))
            else:
                o = np.asarray(odor, dtype=np.float32)
                tgt = self.ORN if len(self.ORN) else self.ALPN
                n = min(len(o), len(tgt))
                idx.append(tgt[:n]); val.append((o[:n]*2.0).astype(np.float32))
        if alpn is not None and len(self.ALPN):
            a = np.asarray(alpn, dtype=np.float32); n = min(len(a), len(self.ALPN))
            idx.append(self.ALPN[:n]); val.append((a[:n]*2.0).astype(np.float32))
        if (odor_left is not None or odor_right is not None):
            if self.mode != "full" or not len(getattr(self, "ORN_L", [])):
                raise ValueError("zapachy boczne wymagaja mode='full' z ORN_L/R (laterality.py)")
            for od, side in ((odor_left, self.ORN_L), (odor_right, self.ORN_R)):
                if od is None:
                    continue
                if isinstance(od, str):
                    si, sv = self._door_lateral(od, side)
                    idx.append(si); val.append(sv)
                else:
                    o = np.asarray(od, dtype=np.float32)
                    n = min(len(o), len(side))
                    idx.append(side[:n]); val.append((o[:n]*2.0).astype(np.float32))
        if mech is not None and len(self.MECH):
            m = np.asarray(mech, dtype=np.float32); n = min(len(m), len(self.MECH))
            idx.append(self.MECH[:n]); val.append((m[:n]*2.0).astype(np.float32))
        if (mech_left is not None or mech_right is not None):
            if self.mode != "full" or not len(getattr(self, "MECH_L", [])):
                raise ValueError("mech boczny wymaga mode='full' z MECH_L/R (laterality.py)")
            for md, side in ((mech_left, self.MECH_L), (mech_right, self.MECH_R)):
                if md is None:
                    continue
                o = np.asarray(md, dtype=np.float32)
                n = min(len(o), len(side))
                idx.append(side[:n]); val.append((o[:n]*2.0).astype(np.float32))
        if image is not None:
            if self.mode != "full" or not len(self.VIS):
                raise ValueError("obraz wymaga mode='full' (retinotopia VIS z pozycji; mb nie ma oczu)")
            v = self._img_to_vec(image)
            vv = np.asarray(v, dtype=np.float32)
            idx.append(self.VIS); val.append((vv[self.VIS_pix]*2.0).astype(np.float32))
            # ruch Reichardta: TYLKO readout (info), bez wstrzykiwania do ME - brak danych o mapowaniu
            small = vv.reshape(64, 64)[:16*4, :16*4].reshape(16, 4, 16, 4).mean(axis=(1, 3)).reshape(-1)
            mot, _ = self._motion_energies(small)
            info["motion"] = mot
        if not idx:
            Iv, Vv = np.zeros(0, np.int32), np.zeros(0, np.float32)
        else:
            Iv, Vv = np.concatenate(idx), np.concatenate(val)
        ck = getattr(self, "_clock", None)
        if ck is not None:
            m = self._clock_state["morning"]
            Iv = np.concatenate([Iv, self._clock_M, self._clock_E])
            Vv = np.concatenate([Vv,
                np.full(len(self._clock_M), self._clock_amp * (2 * m - 1), np.float32),
                np.full(len(self._clock_E), self._clock_amp * (1 - 2 * m), np.float32)])
        if self.asleep and self._auto_sleep is not None:
            Vv = (Vv * self._auto_sleep["gate"]).astype(np.float32)  # prog pobudzenia w gore
        return Iv, Vv, info

    def enable_auto_sleep(self, k_wake=0.05, thr_hi=1.0, thr_lo=0.3, night_lt=0.25,
                          crit_mult=2.0, gate=0.2, dose=0.002):
        """Wlacza sen autonomiczny (dwuprocesowy): S rosnie z aktywnoscia KC (czuwanie),
        spada we snie; sen gdy S>thr_hi i (noc LUB S>crit_mult*thr_hi); pobudka S<thr_lo.
        W snie: drive bramkowany (gate) + mikro-dawka SHY (dose) na krok. Zwraca params."""
        self._auto_sleep = dict(k_wake=k_wake, thr_hi=thr_hi, thr_lo=thr_lo,
                                night_lt=night_lt, crit_mult=crit_mult, gate=gate, dose=dose)
        return self._auto_sleep

    def enable_clock(self, amp=0.5):
        """Wlacza sztuczny TTFL (clock.py, Goldbeter): 1 krok API = 1h.
        Sygnal fazy idzie na WLASCIWE neurony: M (s-LNv/l-LNv, poranne) vs
        E (LNd/DN1, wieczorne), bipolarnie +-amp. Bramka C snu przelacza sie
        ze swiatla na faze zegara (noc = wysoki P2/TIM)."""
        import pandas as pd, re
        from clock import TTFL
        comp = pd.read_csv(f"{self.path}/Completeness_783.csv")
        fly_ids = comp[comp.columns[0]].values.astype(np.int64)
        f2i = {int(f): i for i, f in enumerate(fly_ids)}
        a = pd.read_csv(f"{self.path}/supp1.tsv", sep="\t", usecols=["root_id", "cell_type"])
        t = a["cell_type"].fillna("")
        m = a[t.str.match(r"s-LNv_[ab]|l-LNv") & a["root_id"].astype(int).isin(f2i)]
        e = a[t.str.match(r"LNd_[abc]|DN1a|DN1pA|DN1pB") & a["root_id"].astype(int).isin(f2i)]
        self._clock_M = np.array(sorted({f2i[int(r)] for r in m["root_id"]}), dtype=np.int32)
        self._clock_E = np.array(sorted({f2i[int(r)] for r in e["root_id"]}), dtype=np.int32)
        self._clock = TTFL()
        self._clock_amp = float(amp)
        self._clock_state = {"morning": 0.5, "night_frac": 0.5}
        return {"M": len(self._clock_M), "E": len(self._clock_E), "amp": amp}

    def _sleep_tick(self, kc_frac):
        """Aktualizacja S i przejscia sen/czuwanie. Zwraca (asleep, info)."""
        p = self._auto_sleep
        if p is None:
            return self.asleep, {}
        if getattr(self, "_clock", None) is not None:
            night = bool(self._clock_state.get("night", self._clock_state["night_frac"] > 0.5))
        else:
            night = self._ambient < p["night_lt"]
        if self.asleep:
            self.sleep_S = max(0.0, self.sleep_S - p["k_wake"] * 0.1)
            if self.sleep_S < p["thr_lo"]:
                self.asleep = False
                self._sleep_log.append(("wake", round(self.sleep_S, 3)))
        else:
            self.sleep_S = self.sleep_S + p["k_wake"] * kc_frac
            if self.sleep_S > p["thr_hi"] and (night or self.sleep_S > p["crit_mult"] * p["thr_hi"]):
                self.asleep = True
                self._sleep_log.append(("sleep", round(self.sleep_S, 3)))
        if self.asleep and p["dose"] > 0:
            self.wM[:] = (self.wM - p["dose"] * (self.wM - self.wM0)).astype(np.float32)
        return self.asleep, {"sleep_S": round(self.sleep_S, 3), "asleep": self.asleep,
                             "night": night}

    def enable_scaling(self, cache="fan_abs.npz"):
        """Skalowanie synaptyczne: ABSOLUTNY fan-in (suma |w|). agg/fan = srednia wazona
        wejsc (kontrakcja, stabilnosc). Bez mnoznikow skali - skala = skala bodzca."""
        import os as _os
        if self.mode == "full" and _os.path.exists(f"{self.path}/{cache}"):
            self._fan = np.load(f"{self.path}/{cache}")["fan"].astype(np.float32)
            return "cached"
        if self.mode == "full":
            import numpy as _np
            fan = _np.zeros(self.N, dtype=_np.float64)
            for b in self._pf.iter_batches(batch_size=2000000,
                    columns=["Postsynaptic_Index", "Excitatory x Connectivity"]):
                _np.add.at(fan, _np.asarray(b["Postsynaptic_Index"]).astype(_np.int32),
                           _np.abs(_np.asarray(b["Excitatory x Connectivity"]).astype(float)))
            fan[fan == 0] = 1.0
            np.savez_compressed(f"{self.path}/{cache}", fan=fan.astype(np.float32))
            self._fan = fan.astype(np.float32)
        else:
            fan = np.zeros(self.N, dtype=np.float64)
            np.add.at(fan, self.post, self.wM.astype(float))
            fan[fan == 0] = 1.0
            self._fan = fan.astype(np.float32)
        return "computed"

    def _forward_pure_mb(self, idx, val, hops=1, thr=0.0):
        if getattr(self, "_fan", None) is None:
            self.enable_scaling()
        a = np.zeros(self.N, dtype=np.float32)
        base = np.zeros(self.N, dtype=np.float32)
        if len(idx):
            np.add.at(base, idx, val)
        a = base.copy()
        sw = self.wM*self.sign
        for _ in range(hops):
            msg = a[self.pre]*sw
            agg = np.zeros(self.N, dtype=np.float32)
            np.add.at(agg, self.post, msg)
            agg /= (self._fan + 1e-6)
            a = np.maximum(0, agg - thr) + base
        return a

    # ---- forward (legacy: losowe projekcje; zachowane tylko jako pure=False) ----
    def _forward_mb(self, idx, val):
        x = self.emb.copy()
        if len(idx): np.add.at(x, idx, np.stack([val]*self.in_dim, axis=1))
        msg = (x[self.pre] @ self.Wmsg) * (self.wM*self.sign)[:, None]
        agg = np.zeros((self.N, self.hid), dtype=np.float32)
        np.add.at(agg, self.post, msg)
        if getattr(self, "_fan", None) is not None:
            agg /= (self._fan[:, None] + 1e-6)
        return np.maximum(0, x @ self.Wself + agg)

    def _forward_pure(self, idx, val, hops=2, thr=0.02):
        """Czysta dyfuzja po prawdziwym grafie: ZERO parametrow losowych.
        a = ReLU(suma_wazona - thr*fan) + base. Prog = ulamek calkowitego drive'u
        (koincydencja jak prog kolca); thr to jedna globalna stala protokolu."""
        import numpy as _np
        if getattr(self, "_fan", None) is None:
            self.enable_scaling()
        a = _np.zeros(self.N, dtype=_np.float32)
        base = _np.zeros(self.N, dtype=_np.float32)
        if len(idx):
            _np.add.at(base, idx, val)
        a = base.copy()  # bodziec = zrodlo pradu (clamp), nie stan poczatkowy
        sw = (self.wM*self.sign).astype(_np.float32)  # TRENOWALNE wagi (nie statyczny parquet)
        CH = 2000000
        for _ in range(hops):
            agg = _np.zeros(self.N, dtype=_np.float32)
            for s in range(0, self.E, CH):
                e = slice(s, min(s+CH, self.E))
                _np.add.at(agg, self.post[e], a[self.pre[e]]*sw[e])
            agg /= (self._fan + 1e-6)
            a = _np.maximum(0, agg - thr) + base
        return a

    def _forward_full(self, idx, val, hops=2):
        import numpy as _np
        x = self.emb.copy()
        if len(idx): _np.add.at(x, idx, _np.stack([val]*self.in_dim, axis=1))
        P = x @ self.Wmsg; agg = _np.zeros((self.N, self.hid), dtype=_np.float32)
        for b in self._pf.iter_batches(batch_size=1000000,
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]):
            pb = _np.asarray(b["Presynaptic_Index"]).astype(_np.int32)
            qb = _np.asarray(b["Postsynaptic_Index"]).astype(_np.int32)
            wb = _np.asarray(b["Excitatory x Connectivity"]).astype(_np.float32)
            _np.add.at(agg, qb, P[pb]*wb[:, None])
        if getattr(self, "_fan", None) is not None:
            agg /= (self._fan[:, None] + 1e-6)
        h1 = _np.maximum(0, x @ self.Wself + agg); h1 /= (h1.mean()+1e-6)
        P2 = h1 @ self.Wmsg2; agg2 = _np.zeros((self.N, self.hid), dtype=_np.float32)
        for b in self._pf.iter_batches(batch_size=1000000,
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]):
            pb = _np.asarray(b["Presynaptic_Index"]).astype(_np.int32)
            qb = _np.asarray(b["Postsynaptic_Index"]).astype(_np.int32)
            wb = _np.asarray(b["Excitatory x Connectivity"]).astype(_np.float32)
            _np.add.at(agg2, qb, P2[pb]*wb[:, None])
        if getattr(self, "_fan", None) is not None:
            agg2 /= (self._fan[:, None] + 1e-6)
        h2 = _np.maximum(0, h1 @ self.Wself2 + agg2); h2 /= (h2.mean()+1e-6)
        if hops <= 2 or not hasattr(self, "Wmsg3"):
            return h2
        P3 = h2 @ self.Wmsg3; agg3 = _np.zeros((self.N, self.hid), dtype=_np.float32)
        for b in self._pf.iter_batches(batch_size=1000000,
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]):
            pb = _np.asarray(b["Presynaptic_Index"]).astype(_np.int32)
            qb = _np.asarray(b["Postsynaptic_Index"]).astype(_np.int32)
            wb = _np.asarray(b["Excitatory x Connectivity"]).astype(_np.float32)
            _np.add.at(agg3, qb, P3[pb]*wb[:, None])
        if getattr(self, "_fan", None) is not None:
            agg3 /= (self._fan[:, None] + 1e-6)
        h3 = _np.maximum(0, h2 @ self.Wself3 + agg3); h3 /= (h3.mean()+1e-6)
        return h3

    def calibrate(self, image=None):
        """Baseline do adaptacji: NEUTRALNA scena (domyslnie szarosc 0.5), nie pustka.
        Jak adaptacja do statystyk tla - baseline musi byc niezerowy."""
        import numpy as _np
        if image is None and self.mode == "full":
            image = _np.full((64, 64), 0.5, dtype=_np.float32)
        o = self.step(image=image)
        self._base = {k: o[k] for k in ("VIS_L", "VIS_R", "ALPN_mean", "MB_app", "MB_avo") if k in o}
        self._base_vis = None
        if self.mode == "full":
            idx, val, _ = self.encode(image=image)
            h0 = self._forward_pure(idx, val, hops=2)
            self._base_vis = h0[self.VIS].astype(np.float64)
        return self._base

    def step(self, image=None, odor=None, mech=None, alpn=None, hops=2, pure=None, thr=0.0,
             odor_left=None, odor_right=None, mech_left=None, mech_right=None):
        if getattr(self, "_clock", None) is not None:
            self._clock_state = self._clock.step(self._ambient)
        idx, val, info = self.encode(image=image, odor=odor, mech=mech, alpn=alpn,
                                     odor_left=odor_left, odor_right=odor_right,
                                     mech_left=mech_left, mech_right=mech_right)
        if pure is None:
            pure = True  # domyslnie: tylko dane, zero losowosci (mb i full)
        if pure:
            h = self._forward_pure(idx, val, hops=hops, thr=thr) if self.mode == "full" \
                else self._forward_pure_mb(idx, val, hops=1, thr=thr)
        else:
            h = self._forward_full(idx, val, hops=hops) if self.mode == "full" else self._forward_mb(idx, val)
        kcs = h[self.KC] if pure else h[self.KC].mean(axis=1)
        m = kcs >= np.sort(kcs)[-max(1, int(len(kcs)*0.05))]
        ks = (kcs*m).astype(np.float32)
        km_w = self.wM[self.km_mask]
        r = np.zeros(len(self.MBON)); np.add.at(r, self.km_mi, ks[self.km_ki]*km_w)
        pos = self._mb_pos
        ai = [pos[int(a)] for a in np.sort(self.approach)]
        vi = [pos[int(a)] for a in np.sort(self.avoid)]
        out = {
            "MB_pref": float(r[ai].mean()-r[vi].mean()),
            "MB_app": float(r[ai].mean()), "MB_avo": float(r[vi].mean()),
            "MBON": r.astype(np.float32),
            "KC_active": int(m.sum()), "KC_overlap_n": None,
            "DAN_pam": float(h[self.dan_pam].mean()) if len(self.dan_pam) else 0.0,
            "DAN_ppl": float(h[self.dan_ppl].mean()) if len(self.dan_ppl) else 0.0,
            "ALPN_mean": float(h[self.ALPN].mean()) if len(self.ALPN) else 0.0,
            "EI_sum": float((self.wM*self.sign).sum()),
        }
        if self.mode == "full":
            out["VIS_mean"] = float(h[self.VIS].mean())
            if len(getattr(self, "ALPN_L", [])) and len(getattr(self, "ALPN_R", [])) \
                    and len(self.ALPN):
                out["ALPN_L"] = float(h[self.ALPN_L].mean())
                out["ALPN_R"] = float(h[self.ALPN_R].mean())
                out["turn_olf"] = float(out["ALPN_L"] - out["ALPN_R"])  # >0 zapach po lewej
            if len(getattr(self, "ORN_L", [])) and len(getattr(self, "ORN_R", [])):
                out["ORN_L"] = float(h[self.ORN_L].mean())
                out["ORN_R"] = float(h[self.ORN_R].mean())
            if len(getattr(self, "VIS_eye", [])) == len(self.VIS):
                out["VIS_L"] = float(h[self.VIS[self.VIS_eye == 0]].mean())
                out["VIS_R"] = float(h[self.VIS[self.VIS_eye == 1]].mean())
            out["MECH_mean"] = float(h[self.MECH].mean())
            if len(getattr(self, "MECH_L", [])) and len(getattr(self, "MECH_R", [])):
                out["MECH_L"] = float(h[self.MECH_L].mean())
                out["MECH_R"] = float(h[self.MECH_R].mean())
            out["ORN_mean"] = float(h[self.ORN].mean())
            eff = np.sort(self.EFFERENT); he = len(eff)//2
            out["motor_pref"] = float(h[eff[:he]].mean()-h[eff[he:]].mean())
            out["EFFERENT"] = h[self.EFFERENT] if pure else h[self.EFFERENT].mean(axis=1).astype(np.float32)
            out["EFFERENT"] = np.asarray(out["EFFERENT"], dtype=np.float32)
            if len(self.DESC_L) and len(self.DESC_R):
                out["DN_L"] = float(h[self.DESC_L].mean()); out["DN_R"] = float(h[self.DESC_R].mean())
                out["turn"] = float(out["DN_L"]-out["DN_R"])  # >0 skret w lewo (konwencja)
            if len(self.ME):
                out["ME_mean"] = float(h[self.ME].mean())
            if "motion" in info:
                out["motion"] = info["motion"]
            if getattr(self, "_base", None):
                for k in ("VIS_L", "VIS_R", "ALPN_mean", "MB_app", "MB_avo"):
                    if k in out and k in self._base and abs(self._base[k]) > 1e-9:
                        out[k+"_dff"] = (out[k]-self._base[k])/abs(self._base[k])
            bv = getattr(self, "_base_vis", None)
            if bv is not None and self.mode == "full":
                # adaptacja per-neuron: srednia z dff neuronow (nie dff ze sredniej)
                floor = 0.01*np.abs(bv).mean() + 1e-9
                for nm, m in (("VIS_L_ad", self.VIS[self.VIS_eye == 0]),
                              ("VIS_R_ad", self.VIS[self.VIS_eye == 1])):
                    b = bv[np.isin(self.VIS, m)]
                    v = h[m].astype(np.float64)
                    out[nm] = float(np.mean((v-b)/np.maximum(np.abs(b), floor)))
        if self._auto_sleep is not None:
            _, sinfo = self._sleep_tick(len(self.KC) and out["KC_active"] / max(1, len(self.KC)))
            out.update(sinfo)
        return out

    def sleep(self, episodes=1, rate=0.02):
        """Sen (homeostaza synaptyczna SHY, Tononi-Cirelli): proporcjonalne skalowanie
        magnitud w dol do baseline (wM0) z zachowaniem wzglednych roznic; znaki Dale'a
        nietkniete (operujemy na |w|); setpointy energetyczne sledza nowy stan - brak
        kompensacji w gore (renorm w train() nie cofa snu). rate to protokol (jak 0.85)."""
        for _ in range(episodes):
            self.wM[:] = (self.wM - rate * (self.wM - self.wM0)).astype(np.float32)
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        self.ref_mb[:] = cur
        if self.mode == "full":
            cur = np.zeros(self.N); np.add.at(cur, self.post, self.wM.astype(float))
            self.ref_in[:] = cur
        return {"episodes": episodes, "rate": rate}

    # ================= STREFA EKSPERYMENTALNA (poza pure) =================
    # Metody x_*: wzrost mozgu i zmiany topologii. WYMAGAJA jawnego wywolania,
    # nigdy nie odpalaja sie same. Kazda zmiana trafia do ksiazki _x_log;
    # x_report() mowi dokladnie jak daleko od v783. Dale nowych krawedzi: +1
    # (cholinergiczne, jak KC) - zalozenie protokolu, odnotowane w ksiedze.
    def _x_edges(self, pa, pb, pw, ps):
        n0 = len(self.pre)
        self.pre = np.concatenate([self.pre, np.asarray(pa, np.int32)])
        self.post = np.concatenate([self.post, np.asarray(pb, np.int32)])
        self.wM = np.concatenate([self.wM, np.asarray(pw, np.float32)])
        self.sign = np.concatenate([self.sign, np.asarray(ps, np.float32)])
        self.wM0 = np.concatenate([self.wM0, np.asarray(pw, np.float32)])
        self.E = len(self.pre)
        if self.mode == "full":
            self.elig = np.concatenate([self.elig, np.zeros(len(self.pre) - n0, np.float32)])
            if getattr(self, "_fan", None) is not None and len(self._fan) < self.N:
                self._fan = np.concatenate([self._fan, np.ones(self.N - len(self._fan), np.float32)])
            fan = np.zeros(self.N, dtype=np.float64)
            np.add.at(fan, self.post[n0:], np.abs(pw).astype(float))
            if getattr(self, "_fan", None) is not None:
                self._fan = (self._fan.astype(np.float64) + fan).astype(np.float32)
            if hasattr(self, "ref_in"):
                if len(self.ref_in) < self.N:
                    self.ref_in = np.concatenate([self.ref_in, np.zeros(self.N - len(self.ref_in))])
                self.ref_in = self.ref_in + fan.astype(np.float32)
        if len(self.emb) < self.N:
            rng = np.random.default_rng(1234)
            self.emb = np.concatenate([self.emb, rng.normal(0, 0.5, size=(self.N - len(self.emb), self.in_dim)).astype(np.float32)])
        return n0

    def _x_rebuild_km(self):
        isK = np.zeros(self.N, bool); isK[self.KC] = True
        isM = np.zeros(self.N, bool); isM[self.MBON] = True
        self.km_mask = isK[self.pre] & isM[self.post]
        kc_rank = {int(gg): i for i, gg in enumerate(self.KC)}
        mb_rank = {int(gg): i for i, gg in enumerate(self.MBON)}
        kp = self.pre[self.km_mask]; qp = self.post[self.km_mask]
        self.km_ki = np.array([kc_rank[int(a)] for a in kp], dtype=np.int32)
        self.km_mi = np.array([mb_rank[int(b)] for b in qp], dtype=np.int32)
        avoid_set = set(int(x) for x in self.avoid)
        self.km_is_avoid = np.isin(qp, list(avoid_set))
        self.km_is_approach = ~self.km_is_avoid
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        self.ref_mb = cur

    def x_grow_kc(self, n=100, seed=0, wscale=0.05, per_kc_in=6, per_kc_out=8):
        """Dodaje n pseudo-KC (ids N..N+n-1): wejscia z losowych ALPN (rozklad wag A->K
        z danych x wscale), wyjscia na losowe MBON (rozklad K->M x wscale), znak +1.
        Male wagi urodzeniowe = brak katastrofy; train() je potem rusza (R-Hebb + DAN)."""
        rng = np.random.default_rng(seed)
        isA = np.isin(self.pre, self.ALPN) & np.isin(self.post, self.KC)
        isKM = self.km_mask.copy()
        dA, dM = self.wM[isA], self.wM[isKM]
        new = np.arange(self.N, self.N + n, dtype=np.int32)
        pa, pb, pw = [], [], []
        for kk in new:
            src = rng.choice(self.ALPN, size=min(per_kc_in, len(self.ALPN)), replace=False)
            pa.extend(src); pb.extend([kk] * len(src))
            pw.extend(rng.choice(dA, size=len(src)) * wscale)
            dst = rng.choice(self.MBON, size=min(per_kc_out, len(self.MBON)), replace=False)
            pa.extend([kk] * len(dst)); pb.extend(dst)
            pw.extend(rng.choice(dM, size=len(dst)) * wscale)
        self.N += n
        self.KC = np.sort(np.concatenate([self.KC, new]))
        n0 = self._x_edges(pa, pb, pw, np.ones(len(pw), np.float32))
        self._x_rebuild_km()
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "grow_kc", "n": int(n), "edges": len(pw), "wscale": wscale,
                            "dale": "+1(cholinergiczne)", "i0": int(n0)})
        return new

    def x_add_edge(self, a, b, w, dale=+1.0):
        """Pojedyncza krawedz (z ksiega)."""
        n0 = self._x_edges([a], [b], [abs(w)], [dale])
        self._x_rebuild_km()
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "add_edge", "a": int(a), "b": int(b), "w": float(w),
                            "dale": float(dale), "i0": int(n0)})
        return n0

    def x_cut_edge(self, i):
        """Funkcjonalne ciecie (waga->0, wpis odwracalny; topologia w ksiedze)."""
        i = int(i)
        old = float(self.wM[i])
        self.wM[i] = 0.0
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "cut_edge", "i": i, "a": int(self.pre[i]), "b": int(self.post[i]),
                            "w_old": old})
        return old

    def x_report(self):
        """Ile dodano/wycieto + zalozenia. Zwraca dict, drukuje."""
        lg = getattr(self, "_x_log", [])
        added = sum(e.get("edges", 1) for e in lg if e["op"] in ("grow_kc", "add_edge"))
        cut = sum(1 for e in lg if e["op"] == "cut_edge")
        grown = sum(e.get("n", 0) for e in lg if e["op"] == "grow_kc")
        rep = {"new_neurons": grown, "added_edges": added, "cut_edges": cut,
               "N": self.N, "E": self.E, "ops": len(lg), "dale_new": "+1(cholinergiczne)"}
        print(f"X-REPORT: +{grown} neuronow, +{added} krawedzi, ~{cut} ciec, N={self.N} E={self.E}", flush=True)
        return rep

    def save_weights(self, path):
        np.savez_compressed(path, wM=self.wM)
        try:
            import json, datetime
            reg = f"{self.path}/weights_registry.json"
            import os as _os
            hist = json.load(open(reg)) if _os.path.exists(reg) else []
            hist.append({"path": path, "mode": self.mode,
                         "time": datetime.datetime.now().isoformat(),
                         "EI": float((self.wM*self.sign).sum())})
            json.dump(hist, open(reg, "w"), indent=1)
        except Exception:
            pass
        return path

    def load_weights(self, path):
        w = np.load(path)["wM"].astype(np.float32)
        assert w.shape == self.wM.shape, (w.shape, self.wM.shape)
        self.wM[:] = w
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        self.ref_mb[:] = cur
        return path

    def train(self, image=None, odor=None, mech=None, alpn=None, reward=0.0, punish=0.0, pure=True, thr=0.0,
              odor_left=None, odor_right=None, mech_left=None, mech_right=None):
        """reward>0 (PAM: slab-avoid) / punish>0 (PPL1: slab-approach). Zwraca step() po nauce."""
        idx, val, _ = self.encode(image=image, odor=odor, mech=mech, alpn=alpn,
                                  odor_left=odor_left, odor_right=odor_right,
                                  mech_left=mech_left, mech_right=mech_right)
        if pure:
            h = self._forward_pure(idx, val, hops=2, thr=thr) if self.mode == "full" \
                else self._forward_pure_mb(idx, val, hops=1, thr=thr)
            kcs = h[self.KC]
        else:
            h = self._forward_full(idx, val) if self.mode == "full" else self._forward_mb(idx, val)
            kcs = h[self.KC].mean(axis=1)
        m = kcs >= np.sort(kcs)[-max(1, int(len(kcs)*0.05))]
        if self.mode == "full" and reward != 0:
            a = h if pure else h.mean(axis=1).astype(np.float32)
            for s in range(0, self.E, self.CH):
                e = slice(s, min(s+self.CH, self.E))
                self.elig[e] = self.elig[e]*0.9 + (a[self.pre[e]]*a[self.post[e]]).astype(np.float32)
            for s in range(0, self.E, self.CH):
                e = slice(s, min(s+self.CH, self.E))
                dw = np.clip(0.002*float(reward)*self.elig[e], -0.02*self.wM[e], 0.02*self.wM[e])
                self.wM[e] = np.clip(self.wM[e]+dw-1e-6, 0.05, 650.0)
            cur = np.zeros(self.N); np.add.at(cur, self.post, self.wM.astype(float))
            sc = np.ones(self.N); nz = cur > 1e-9
            sc[nz] = np.clip(self.ref_in[nz]/cur[nz], 0.95, 1.05)
            for s in range(0, self.E, self.CH):
                e = slice(s, min(s+self.CH, self.E))
                self.wM[e] = self.wM[e]*sc[self.post[e]]
        if reward > 0:
            sel = self.km_is_avoid & m[self.km_ki]
            self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*0.85, self.wM[self.km_mask])
        if punish > 0:
            sel = self.km_is_approach & m[self.km_ki]
            self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*0.85, self.wM[self.km_mask])
        self.wM[self.km_mask] = np.maximum(self.wM[self.km_mask], 0.05)
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        sc = np.ones(len(self.MBON)); nz = cur > 1e-9
        sc[nz] = np.clip(self.ref_mb[nz]/cur[nz], 0.9, 1.1)
        self.wM[self.km_mask] = (self.wM[self.km_mask]*sc[self.km_mi]).astype(np.float32)
        return self.step(image=image, odor=odor, mech=mech, alpn=alpn,
                         odor_left=odor_left, odor_right=odor_right,
                         mech_left=mech_left, mech_right=mech_right)
