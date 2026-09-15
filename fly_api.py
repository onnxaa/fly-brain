"""Full fly-brain API (numpy, no torch).

DATA: FlyWire FAFB v783 (Dorkenwald et al. Nature 2024, CC BY-NC 4.0 - NON-COMMERCIAL USE ONLY),
  flyconnectome annotations (supp1.tsv), DoOR 2.0 profiles (Munch & Galizia 2016),
  GRN lists (Shiu et al. Nature 2024), MBON valences (Aso et al. eLife 2014).
RULE: zero parameters outside the data - topology, Dale signs and |w| from the connectome;
  the only constants are experiment protocol (drive, top-k 5%, anti-Hebb, thr).
Pure mode (default) = clean diffusion; legacy rate (pure=False) kept for comparison.

Sensory inputs (any combination at once):
  image   : HxW float 0..1 matrix (grayscale) -> R photoreceptors (real front end:
            each R1-6/R7/R8 samples the image at its own receptive field from
            real pos_x/pos_y, bilinear + Gaussian RF; no pixel grid)
   odor    : (n_ORN,) float 0..1 vector OR 'A'/'B' word (ORN halves, as in tests)
   odor_left/odor_right : lateral DoOR odor (name) OR per-side vector
             (full + ORN_L/R from laterality.py; adds up with odor)
             Lateral outputs: ALPN_L/ALPN_R, ORN_L/R, turn_olf = ALPN_L-ALPN_R
   mech_left/mech_right : lateral touch, per-side vectors (full + MECH_L/R);
             MECH_L/R outputs
   mech    : (n_MECH,) float 0..1 vector (touch/vibration)
   alpn    : direct glomerular drive (n_ALPN,) - bypasses ORN (1-hop to KC in mb mode)
   dan_rew : float 0..1 (PAM, reward), dan_pun : float 0..1 (PPL1, punishment)

Outputs (dict from step()/train()):
  MB_pref, MB_app, MB_avo, MBON (96,), DAN_pam, DAN_ppl, KC_active, KC_overlap_info,
  motor_pref, motor_A, motor_B, EFFERENT (1481,), ALPN_mean, VIS_mean, MECH_mean, ORN_mean, EI_sum

Modes:
  mode='mb'   : MB circuit 6289/548k, 1-hop, ~1-2 s/step (default, interactive)
  mode='full' : whole brain 138639/15M, 2-hop + gain control, ~20 s/step (parquet stream)

Learning: train() does R-Hebb (full: whole graph +-2% + renorm; mb: K->M only)
  + DAN anti-Hebb on K->M (reward weakens avoid, punishment weakens approach).
Sleep: sleep() does homeostatic downscaling to baseline (SHY) - forgetting.
Topology and Dale signs frozen - brain identity preserved (see validate_flyness.py).
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
            # mb mode has no VIS/MECH/ORN/EFFERENT -> map sensors onto ALPN:
            # odor->ALPN (direct), image makes no sense in mb, so we project it
            # onto KC via a random projection (optics placeholder in MB).
            self.VIS = np.array([], dtype=np.int32); self.MECH = np.array([], dtype=np.int32)
            self.ORN = np.array([], dtype=np.int32); self.EFFERENT = np.array([], dtype=np.int32)
            self._img_proj = rng.normal(0, 1.0, size=(IMG_N, min(256, len(self.KC)))).astype(np.float32)
            self._img_kc = np.sort(self.KC)[:self._img_proj.shape[1]]
        else:
            r = np.load(f"{path}/roles_full.npz")
            self.N = int(r["N"][0])
            self.ORN, self.MECH, self.VIS = r["ORN"], r["MECH"], r["VIS"]
            self.ALPN, self.EFFERENT = r["ALPN"], r["EFFERENT"]
            # olfactory laterality (laterality.py, side from supp1.tsv; missing = empty)
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
            self._last_small = None  # for the motion detector (T4/T5-like)
            mb = np.load(f"{path}/mb_circuit.npz"); mb_fly = mb["flywire_ids"]
            import pandas as pd
            comp = pd.read_csv(f"{path}/Completeness_783.csv")
            fly_ids = comp[comp.columns[0]].values.astype(np.int64)
            fly2idx = {int(f): i for i, f in enumerate(fly_ids)}
            # real photoreceptor front end: R1-6/R7/R8 sample the image at their
            # own receptive fields from real pos_x/pos_y (rank-normalized to
            # [0,1], robust like qcut but continuous - no 64x64 grid).
            if "R" in r and "R_cx" in r and "R_cy" in r:
                self.R = np.asarray(r["R"], dtype=np.int32)
                self.R_cx = np.asarray(r["R_cx"], dtype=np.float32)
                self.R_cy = np.asarray(r["R_cy"], dtype=np.float32)
            else:
                try:
                    _a = pd.read_csv(f"{path}/supp1.tsv", sep="\t",
                                     usecols=["root_id", "cell_type", "pos_x", "pos_y"])
                    _rm = _a[_a["cell_type"].isin(["R1-6", "R7", "R8"]) &
                              (_a["root_id"].astype(int).isin(fly2idx))].copy()
                    self.R = np.array([fly2idx[int(x)] for x in _rm["root_id"]], dtype=np.int32)
                    _rpx = _rm["pos_x"].values.astype(float)
                    _rpy = _rm["pos_y"].values.astype(float)
                    _nR = len(self.R)
                    _ox = np.argsort(_rpx, kind="stable")
                    _oy = np.argsort(_rpy, kind="stable")
                    _cx = np.empty(_nR, dtype=np.float64)
                    _cy = np.empty(_nR, dtype=np.float64)
                    _cx[_ox] = np.arange(_nR) / max(1, _nR - 1)
                    _cy[_oy] = np.arange(_nR) / max(1, _nR - 1)
                    self.R_cx = _cx.astype(np.float32)
                    self.R_cy = _cy.astype(np.float32)
                    try:
                        _d = dict(r)
                        _d["R"] = self.R
                        _d["R_cx"] = self.R_cx
                        _d["R_cy"] = self.R_cy
                        np.savez(f"{path}/roles_full.npz", **_d)
                    except Exception:
                        pass
                except Exception:
                    self.R = np.zeros(0, np.int32)
                    self.R_cx = np.zeros(0, np.float32)
                    self.R_cy = np.zeros(0, np.float32)
            # legacy discrete bins (see_lif 64x64) kept for reference only, unused.
            if "R_rx" in r:
                self.R_rx = np.asarray(r["R_rx"])
            else:
                self.R_rx = None
            if "R_ry" in r:
                self.R_ry = np.asarray(r["R_ry"])
            else:
                self.R_ry = None
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
        # shared: K->M weights + embeddings
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
        qp_all = self.post[self.km_mask]  # once (was: mask copy in each of 62k iterations!)
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
        self.wM0 = self.wM.copy()  # sleep baseline (SHY): only learning deviations drift
        self.CH = 2000000
        # autonomous sleep (Borbely two-process model): S=homeostat from brain activity,
        # C=light gate (zeitgeber). Scalar S like elig (extra-neural trace); the sleep/wake
        # decision follows brain state + environment, not the experimenter. Off by default.
        self._auto_sleep = None
        self.sleep_S = 0.0
        self.asleep = False
        self._ambient = 0.5
        self._sleep_log = []
        # readout depth (protocol constant, pure-safe: changes no weights).
        # full default 2 (ORN->ALPN->KC); 3-4 pulls vision deeper toward KC
        # (R reaches 5% of KC in 2 hops, more in 3-4) but noisier/attenuated.
        # mb default 1 (MBON readout is the analytic 2nd hop).
        self._hops = 1 if mode == "mb" else 2

        # activation: 'relu' (fast legacy) or 'lif' (saturating LIF-shaped rate).
        # 'lif' is NOT the exact log-form steady state: measured on our
        # fan-scaled drives it sits in the log singularity at rheobase
        # (infinite slope -> every cell fires ~5-15Hz, contrast crushed).
        # Instead: noisy-LIF/saturating rectifier f(x)=SAT*(1-exp(-x/SAT)),
        # linear with slope 1 at rheobase (matches relu on weak drives, so all
        # validations transfer approx), saturating like 1/t_ref on strong
        # drives (relu's unbounded growth is the actual incorrect feature).
        # Shiu constants behind the shape: t_mbr 20ms, t_rfc 2.2ms,
        # Vth-Vrst 7mV. Full spike timing stays in see_lif.py/Brian.
        self._act = "relu"
        self._lif_sat = 2.0
        self._spike_T = 200
        self._spike_seed = 7
        self._spike_wdrv = 0.275 * 250.0  # Brian f_poi=250 (see calibration note)
        self._spike_ainc = 4.0  # SFA mV/spike, tau 100ms
        self._spike_rmax = 150.0  # Poisson Hz at val=2.0 (test_lif R150)
        # stabilizers are mode-aware: mb = test_lif v2 1:1 (validated KC 5%),
        # full recurrent loops need them (measured avalanche without).
        self._spike_adapt = 1.0 if mode == "mb" else 0.2  # drive floor
        if mode == "mb":
            self._spike_ainc = 0.0
        self._spike_burn = 0 if mode == "mb" else 50  # onset ms not counted
        self._spike_cache = {}

    def set_hops(self, hops):
        """Default diffusion depth for step/train/teach/probes (None arg = this).
        full: 2 clean olfactory, 3-4 deep multimodal (vision reaches KC);
        mb: 1 by design (2 experimental). Returns the new default."""
        h = int(hops)
        if h < 1 or h > 6:
            raise ValueError("hops must be 1..6 (3-4 = deep/noisy, 5-6 = flood risk)")
        self._hops = h
        return self._hops

    def get_hops(self):
        """Current default readout depth."""
        return int(getattr(self, "_hops", 2 if self.mode == "full" else 1))

    def set_activation(self, name, sat=None, Tms=None, seed=None, wdrv=None, ainc=None, rmax=None,
                       adapt=None, burn=None):
        """Activation mode for the forward passes.

        'relu' (default): f(x)=max(0,x) - fast, legacy, all PASS numbers.
        'lif': f(x)=SAT*(1-exp(-x/SAT)) for x>0 else 0 - saturating LIF-shaped
        rate (see __init__ note on why not the exact log-form). Pure-safe
        protocol constant (no weights change), but absolute readout scales
        compress on strong drives - compare within one mode.
        'spike': full LIF spiking simulation (test_lif v2 equations 1:1 -
        g-reset, 2-step refractory/delay, signed weights, spiking APL x4,
        graded early vision with tonic release in full mode). step() returns
        the same dict in Hz. Slow: mb ~seconds/200ms, full ~1min/200ms.
        sat: optionally set the lif saturation scale (drive units, >0).
        Tms/seed: spike window ms and RNG seed (defaults 200/7).
        wdrv: Poisson drive weight mV (default Brian 68.75 = 0.275*250).
        ainc: SFA increment mV/spike (default 4.0, tau 100ms).
        rmax: Poisson ceiling Hz at val=2.0 (default 150 = test_lif R150).
        adapt: sensory-adaptation floor 0..1 (mb default 1=off, full 0.2).
        burn: onset ms excluded from rate counts (mb 0, full 50).
        Returns the new mode."""
        if name not in ("relu", "lif", "spike"):
            raise ValueError("activation must be 'relu', 'lif' or 'spike'")
        self._act = name
        if sat is not None:
            s = float(sat)
            if s <= 0:
                raise ValueError("sat must be >0")
            self._lif_sat = s
        if Tms is not None:
            t = int(Tms)
            if t < 10 or t > 5000:
                raise ValueError("Tms must be 10..5000")
            self._spike_T = t
        if seed is not None:
            self._spike_seed = int(seed)
        if wdrv is not None:
            w = float(wdrv)
            if w <= 0 or w > 200:
                raise ValueError("wdrv must be 0..200mV")
            self._spike_wdrv = w
        if ainc is not None:
            a = float(ainc)
            if a < 0 or a > 50:
                raise ValueError("ainc must be 0..50mV")
            self._spike_ainc = a
        if rmax is not None:
            r = float(rmax)
            if r <= 0 or r > 500:
                raise ValueError("rmax must be 0..500Hz")
            self._spike_rmax = r
        if adapt is not None:
            a = float(adapt)
            if a < 0 or a > 1:
                raise ValueError("adapt must be 0..1")
            self._spike_adapt = a
        if burn is not None:
            b = int(burn)
            if b < 0 or b > 1000:
                raise ValueError("burn must be 0..1000ms")
            self._spike_burn = b
        return self._act

    def get_activation(self):
        """Current activation mode."""
        return str(getattr(self, "_act", "relu"))

    def set_std(self, alpha=0.0, tau=25.0):
        """Short-term depression of input drive (optional, OFF by default).

        Presynaptic-style: each driven neuron's gain g (init 1) scales its
        outgoing drive val*g, then depresses g *= (1-alpha*a) with activity
        a = clip(val/2, 0, 1); every step all gains recover
        g += (1-g)/tau. Repeated strong drive habituates, novel pathways
        stay fresh (dishabituation by anatomy), silence restores.
        Applies in step() only (behavioral expression); train() /
        x_teach_output() use full drive. State is transient (not saved by
        save_weights). PROTOCOL, not data: tau/alpha are free constants
        (units: trials; 1 step = 1 trial), default off so every published
        PASS number is unaffected. Returns the new setting."""
        a = float(alpha)
        t = float(tau)
        if a < 0 or a > 1:
            raise ValueError("std alpha must be 0..1 (0=off)")
        if t < 1:
            raise ValueError("std tau must be >=1 trial")
        self._std = {"alpha": a, "tau": t}
        if a > 0 and getattr(self, "_std_g", None) is None:
            self._std_g = np.ones(int(self.N), dtype=np.float32)
        return dict(self._std)

    def get_std(self):
        """Current short-term-depression setting (default off)."""
        d = getattr(self, "_std", None)
        if d is None:
            return {"alpha": 0.0, "tau": 25.0}
        return dict(d)

    def _apply_std(self, idx, val, alpha, tau):
        n = int(self.N)
        g = getattr(self, "_std_g", None)
        if g is None or len(g) != n:
            ng = np.ones(n, dtype=np.float32)
            if g is not None:
                ng[:len(g)] = np.asarray(g, dtype=np.float32)
            g = ng
            self._std_g = g
        g += (1.0 - g) / float(tau)
        out_v = []
        for ik, vk in zip(idx, val):
            ik = np.asarray(ik, dtype=np.int64)
            vk = np.asarray(vk, dtype=np.float32)
            a = np.clip(vk / 2.0, 0.0, 1.0).astype(np.float32)
            out_v.append((vk * g[ik]).astype(np.float32))
            g[ik] = np.clip(g[ik] * (1.0 - alpha * a), 0.0, 1.0)
        return idx, out_v

    def _activate(self, x):
        """Elementwise f-I curve on (drive - thr)."""
        if getattr(self, "_act", "relu") == "lif":
            s = float(getattr(self, "_lif_sat", 2.0))
            x = np.asarray(x)
            out = np.zeros_like(x, dtype=np.float32)
            m = x > 0
            out[m] = (s * (1.0 - np.exp(-x[m] / s))).astype(np.float32)
            return out
        return np.maximum(0, x)

    # ---- full spiking (test_lif v2 equations 1:1) ----
    def _spike_ensure_caches(self):
        """APL indices / graded mask / mb APL-extension pieces. Built once."""
        c = getattr(self, "_spike_cache", {})
        if self.mode == "full":
            if "apl" not in c:
                import pandas as _pd
                comp = _pd.read_csv(f"{self.path}/Completeness_783.csv")
                fly_ids = comp[comp.columns[0]].values.astype(np.int64)
                f2i = {int(f): i for i, f in enumerate(fly_ids)}
                a = _pd.read_csv(f"{self.path}/supp1.tsv", sep="\t",
                                 usecols=["root_id", "cell_type"])
                c["apl"] = np.array(
                    sorted({f2i[int(r)] for r in
                            a[a["cell_type"] == "APL"]["root_id"]
                            if int(r) in f2i}), dtype=np.int32)
            if "gset" not in c:
                import pandas as _pd, re as _re
                comp = _pd.read_csv(f"{self.path}/Completeness_783.csv")
                fly_ids = comp[comp.columns[0]].values.astype(np.int64)
                f2i = {int(f): i for i, f in enumerate(fly_ids)}
                a = _pd.read_csv(f"{self.path}/supp1.tsv", sep="\t",
                                 usecols=["root_id", "cell_type"])
                typ = dict(zip(a["root_id"].astype(int), a["cell_type"]))
                gpat = _re.compile(
                    r"^(R1-6|R7|R8|Lai|L[1-5]|Dm\d+|DmDRA\d+|Mi[149]|"
                    r"Tm[1-49]|Tm20|C[23])$")
                c["gset"] = np.array(
                    sorted({f2i[int(r)] for r, t in typ.items()
                            if isinstance(t, str) and gpat.match(t)
                            and int(r) in f2i}), dtype=np.int32)
                self._spike_cache = c
            return c["apl"], c["gset"], None
        # mb: 2 spiking APL nodes from real KC<->APL weights (test_lif v2).
        if "ext" not in c:
            import pandas as _pd
            d = np.load(f"{self.path}/mb_circuit.npz")
            comp = _pd.read_csv(f"{self.path}/Completeness_783.csv")
            fly = comp[comp.columns[0]].values.astype("int64")
            f2i = {int(f): i for i, f in enumerate(fly)}
            sup = _pd.read_csv(f"{self.path}/supp1.tsv", sep="\t",
                               usecols=["root_id", "cell_type"])
            apl_g = [f2i[int(r)] for r in sup[sup["cell_type"] == "APL"]["root_id"]
                     if int(r) in f2i]
            con = _pd.read_parquet(
                f"{self.path}/Connectivity_783.parquet",
                columns=["Presynaptic_Index", "Postsynaptic_Index",
                         "Excitatory x Connectivity"])
            loc2glob = np.array([f2i[int(f)] for f in d["flywire_ids"]],
                                dtype=np.int32)
            glob2loc = {int(gg): int(ll) for ll, gg in enumerate(loc2glob)}
            apl_id = {g: k for k, g in enumerate(apl_g)}
            k2a = con[con["Postsynaptic_Index"].isin(apl_g) &
                      con["Presynaptic_Index"].isin(glob2loc)]
            a2k = con[con["Presynaptic_Index"].isin(apl_g) &
                      con["Postsynaptic_Index"].isin(glob2loc)]
            c["ext"] = dict(
                k2a_pre=np.array([glob2loc[int(p)] for p in
                                  k2a["Presynaptic_Index"].values], dtype=np.int32),
                k2a_w=np.abs(k2a["Excitatory x Connectivity"].values.astype(np.float64)),
                k2a_apl=np.array([apl_id[int(q)] for q in
                                  k2a["Postsynaptic_Index"].values], dtype=np.int32),
                a2k_post=np.array([glob2loc[int(q)] for q in
                                   a2k["Postsynaptic_Index"].values], dtype=np.int32),
                a2k_w=np.abs(a2k["Excitatory x Connectivity"].values.astype(np.float64)),
                a2k_apl=np.array([apl_id[int(p)] for p in
                                  a2k["Presynaptic_Index"].values], dtype=np.int32))
            self._spike_cache = c
            del con, comp, sup
        return None, None, c["ext"]

    def _forward_spike(self, idx, val, Tms=None, seed=None):
        """LIF spiking forward (test_lif v2 1:1): Poisson drive from encode
        (rate = val*75Hz, so val 2.0 = 150Hz like DoOR R150 / RMAX150),
        g-reset, 2-step refractory/delay, signed weights, APL x4, graded
        early vision (tonic release) in full mode. Returns Hz activity
        (spikes + release events). X-zone neurons spike with default thr."""
        from collections import deque
        T = int(Tms) if Tms is not None else int(getattr(self, "_spike_T", 200))
        seed = self._spike_seed if seed is None else int(seed)
        V0, VRST, VTH = -52.0, -52.0, -45.0
        DT = 1.0
        DEC_G = float(np.exp(-DT / 5.0))
        K_MBR = DT / 20.0
        W_SYN, APL_F, KREL = 0.275, 4.0, 20.0
        WDRV = float(getattr(self, "_spike_wdrv", 0.275 * 250.0))
        apl, gset, ext = self._spike_ensure_caches()
        # No synaptic scaling (keeps Shiu 1:1 raw weights): recurrent-loop
        # stability comes from spike-frequency adaptation instead (mAHP-like
        # current, biologically real; the MB-circuit regime it was validated
        # in has no recurrent ALPN loops, the full brain does - measured:
        # raw drive avalanches to ALPN 111Hz/KC 77% without it).
        if ext is not None:  # mb: 2 APL nodes above current N (no collision)
            A0, A1, nN = self.N, self.N + 1, self.N + 2
            pre = np.concatenate([self.pre, ext["k2a_pre"], A0 + ext["a2k_apl"]])
            post = np.concatenate([self.post, A0 + ext["k2a_apl"], ext["a2k_post"]])
            w = np.concatenate([self.wM * self.sign, ext["k2a_w"], -ext["a2k_w"]])
            apl_idx, graded = np.array([A0, A1], dtype=np.int32), np.zeros(nN, bool)
        else:
            pre, post = self.pre, self.post
            w = (self.wM * self.sign).astype(np.float64)
            nN = self.N
            apl_idx = np.asarray(apl, dtype=np.int32)
            graded = np.zeros(nN, bool)
            gg_idx = np.asarray(gset, dtype=np.int32)
            gg_idx = gg_idx[gg_idx < nN]
            graded[gg_idx] = True
        v = np.full(nN, V0, np.float64)
        vth = np.full(nN, VTH, np.float64)
        if len(apl_idx):
            vth[apl_idx[apl_idx < nN]] = V0 + APL_F * (VTH - V0)
        gg = np.zeros(nN, np.float64)
        refr = np.zeros(nN, np.int16)
        adapt = np.zeros(nN, np.float64)  # SFA current (mAHP-like)
        DEC_A = float(np.exp(-DT / 100.0))
        A_INC = float(getattr(self, "_spike_ainc", 4.0))  # mV per spike
        nsp = np.zeros(nN, np.int32)
        nrel = np.zeros(nN, np.int32)
        didx = np.asarray(idx, dtype=np.int32)
        didx = didx[didx < nN]
        dval = np.asarray(val, dtype=np.float64)[:len(didx)] if len(didx) else np.zeros(0)
        # Poisson drive: rate = val/2*rmax (val 2.0 = rmax, test_lif R150).
        dp = dval * float(getattr(self, "_spike_rmax", 150.0)) / 2.0 * DT / 1000.0
        G = np.where(graded)[0]
        dq = deque([np.zeros(nN, np.float64) for _ in range(2)])
        rng = np.random.default_rng(seed)
        # sensory adaptation (ORN/R adapt in ~100ms; without it sustained
        # 150Hz Poisson piles 50mV+ into g and the brain saturates or dies -
        # measured cliff: wdrv 27.5 -> KC 77%, 13.75 -> KC 0%).
        # Rates count after burn-in (onset transient discarded, standard).
        AD_TAU = 50.0
        AD_FLOOR = float(getattr(self, "_spike_adapt", 0.2))
        BURN = int(getattr(self, "_spike_burn", 50))
        for t in range(T):
            gg *= DEC_G
            adapt *= DEC_A
            gg += dq.popleft()
            dq.append(np.zeros(nN, np.float64))
            if len(didx):
                ad = AD_FLOOR + (1.0 - AD_FLOOR) * float(np.exp(-t / AD_TAU))
                fire = rng.random(len(didx)) < np.abs(dp) * ad
                if fire.any():
                    fi = didx[fire]
                    gg[fi[dp[fire] > 0]] += WDRV
                    gg[fi[dp[fire] < 0]] -= WDRV
            awake = (refr == 0) & (~graded)
            v[awake] += K_MBR * (V0 - v[awake] + gg[awake] - adapt[awake])
            v[graded] += K_MBR * (V0 - v[graded] + gg[graded])
            refr[refr > 0] -= 1
            sp = np.where(awake & (v >= vth))[0]
            rel = G[rng.random(len(G)) <
                    np.clip(v[G] - V0, 0, None) * KREL * DT / 1000.0] if len(G) else G[:0]
            ev = np.concatenate([sp, rel]) if len(rel) else sp
            if t < BURN:
                # onset transient: propagate spikes (network state) but do not
                # count them toward rates.
                if len(sp):
                    v[sp] = VRST
                    gg[sp] = 0.0
                    adapt[sp] += A_INC
                    refr[sp] = 2
                if len(ev):
                    m = np.isin(pre, ev)
                    if m.any():
                        dq[-1] += np.bincount(post[m], weights=(w[m] * W_SYN).astype(np.float64),
                                              minlength=nN)
                continue
            if len(sp):
                v[sp] = VRST
                gg[sp] = 0.0
                adapt[sp] += A_INC
                refr[sp] = 2
                nsp[sp] += 1
            if len(rel):
                nrel[rel] += 1
            if len(ev):
                m = np.isin(pre, ev)
                if m.any():
                    dq[-1] += np.bincount(post[m], weights=(w[m] * W_SYN).astype(np.float64),
                                          minlength=nN)
        _win = max(1, T - BURN)
        return (nsp + nrel) / _win * 1000.0

    # ---- sensors ----
    def _gray(self, image):
        a = np.asarray(image, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        return np.clip(a, 0, 1)

    def _sample_R(self, gray):
        """Per-R sampling: each R cell reads the image at its own (cx, cy) with
        bilinear interpolation over a 3x3 Gaussian-blurred frame (RF ~1 px).
        No pixel grid, no qcut bins - rank order from anatomy is the only map."""
        a = np.asarray(gray, dtype=np.float32)
        H, W = a.shape
        # 3x3 Gaussian RF [[1,2,1],[2,4,2],[1,2,1]]/16, edge-replicated pad.
        ap = np.pad(a, 1, mode="edge")
        b = (4 * ap[1:-1, 1:-1] + 2 * (ap[:-2, 1:-1] + ap[2:, 1:-1] +
             ap[1:-1, :-2] + ap[1:-1, 2:]) +
             ap[:-2, :-2] + ap[:-2, 2:] + ap[2:, :-2] + ap[2:, 2:]) / 16.0
        xs = self.R_cx.astype(np.float64) * (W - 1)
        ys = self.R_cy.astype(np.float64) * (H - 1)
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.clip(x0, 0, max(0, W - 1))
        y0 = np.clip(y0, 0, max(0, H - 1))
        x1 = np.minimum(x0 + 1, max(0, W - 1))
        y1 = np.minimum(y0 + 1, max(0, H - 1))
        fx = (xs - x0).astype(np.float32)
        fy = (ys - y0).astype(np.float32)
        Ia = b[y0, x0]
        Ib = b[y0, x1]
        Ic = b[y1, x0]
        Id = b[y1, x1]
        return np.clip(Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy) +
                       Ic * (1 - fx) * fy + Id * fx * fy, 0, 1)

    def _small16(self, gray):
        """16x16 frame for the Reichardt readout, straight from the input."""
        a = np.asarray(gray, dtype=np.float32)
        H, W = a.shape
        bh, bw = max(1, H // 16), max(1, W // 16)
        return a[:16 * bh, :16 * bw].reshape(16, bh, 16, bw).mean(axis=(1, 3)).reshape(-1)

    def _img_to_vec(self, image):
        # Kept for backward compatibility: 64x64 flat vector (motion path only).
        a = self._gray(image)
        H, W = a.shape
        bh, bw = max(1, H // 64), max(1, W // 64)
        small = a[:64 * bh, :64 * bw].reshape(64, bh, 64, bw).mean(axis=(1, 3))
        return np.clip(small.reshape(-1), 0, 1)

    def _motion_energies(self, small):
        """Reichardt (T4/T5-like): 4 directions from a 16x16 frame pair. Returns dict R/L/U/D >=0."""
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
        """DoOR profile slice for one side (L/R): mask idx to ORN_L/R. Full mode only."""
        import os as _os4
        _door = f"{self.path}/door_odors.npz"
        if not _os4.path.exists(_door) or odor not in (
                "geosmin", "co2", "hexanone3", "methyl_salicylate", "butanedione", "ethyl_hexanoate"):
            raise ValueError(f"lateral odor '{odor}': DoOR odors only (door_odors.npz)")
        _dd = np.load(_door)
        oi, ov = _dd[odor + "_idx"], (_dd[odor + "_val"] * 2.0).astype(np.float32)
        keep = np.isin(oi, side)
        return oi[keep], ov[keep]

    def encode(self, image=None, odor=None, mech=None, alpn=None, odor_left=None, odor_right=None,
               mech_left=None, mech_right=None):
        """Returns (idx, val, info): info has 'motion' when motion was computed."""
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
                        raise ValueError("GRN tastes require mode='full'")
                    _tt = np.load(_taste)
                    sel = _tt[odor+"_idx"]
                    idx.append(sel); val.append(np.full(len(sel), 2.0, np.float32))
                elif _os3.path.exists(_door) and odor in (
                        "geosmin", "co2", "hexanone3", "methyl_salicylate", "butanedione", "ethyl_hexanoate"):
                    if self.mode != "full":
                        raise ValueError(f"odor '{odor}' (DoOR, ORN) requires mode='full'; in mb use 'A'/'B' or a vector")
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
                raise ValueError("lateral odors require mode='full' with ORN_L/R (laterality.py)")
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
                raise ValueError("lateral mech requires mode='full' with MECH_L/R (laterality.py)")
            for md, side in ((mech_left, self.MECH_L), (mech_right, self.MECH_R)):
                if md is None:
                    continue
                o = np.asarray(md, dtype=np.float32)
                n = min(len(o), len(side))
                idx.append(side[:n]); val.append((o[:n]*2.0).astype(np.float32))
        if image is not None:
            if self.mode != "full" or not len(getattr(self, "R", [])):
                raise ValueError("images require mode='full' with R photoreceptors (see_lif mapping; mb has no eyes)")
            # Phototransduction only: each R cell samples the image at its own
            # receptive field (bilinear + Gaussian RF). No grid, no bins.
            g = self._gray(image)
            idx.append(self.R)
            val.append((self._sample_R(g) * 2.0).astype(np.float32))
            # Reichardt motion: readout ONLY (info), never injected into ME - no mapping data
            mot, _ = self._motion_energies(self._small16(g))
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
            Vv = (Vv * self._auto_sleep["gate"]).astype(np.float32)  # arousal threshold up
        return Iv, Vv, info

    def enable_auto_sleep(self, k_wake=0.05, thr_hi=1.0, thr_lo=0.3, night_lt=0.25,
                          crit_mult=2.0, gate=0.2, dose=0.002):
        """Enables autonomous sleep (two-process): S rises with KC activity (wake),
        falls in sleep; sleep when S>thr_hi and (night OR S>crit_mult*thr_hi); wake S<thr_lo.
        In sleep: gated drive (gate) + SHY micro-dose (dose) per step. Returns params."""
        self._auto_sleep = dict(k_wake=k_wake, thr_hi=thr_hi, thr_lo=thr_lo,
                                night_lt=night_lt, crit_mult=crit_mult, gate=gate, dose=dose)
        return self._auto_sleep

    def enable_clock(self, amp=0.5):
        """Enables the artificial TTFL (clock.py, Goldbeter): time flows ONLY through
        tick_clock(hours) - behavior steps never advance the clock (scale separation).
        Phase signal goes to the CORRECT neurons: M (s-LNv/l-LNv, morning) vs
        E (LNd/DN1, evening), bipolar +-amp. Sleep gate C switches from light
        to clock phase (night = high P2/TIM)."""
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
        self._clock_state = {"morning": 0.5, "night_frac": 0.5, "night": False}
        return {"M": len(self._clock_M), "E": len(self._clock_E), "amp": amp}

    def tick_clock(self, hours=1.0, light=None):
        """Advances the clock by hours h (ALL circadian time; behavior steps never touch it).
        light=None: uses current _ambient; result: phase + M/E drive refreshed at next encode."""
        if getattr(self, "_clock", None) is None:
            raise ValueError("enable_clock() first")
        if light is None:
            light = self._ambient
        n = max(1, int(round(hours)))
        for _ in range(n):
            self._clock_state = self._clock.step(light)
        return dict(self._clock_state)

    def _sleep_tick(self, kc_frac):
        """Sleep-pressure S update + sleep/wake transitions. Returns (asleep, info)."""
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
        """Synaptic scaling: ABSOLUTE fan-in (sum of |w|). agg/fan = weighted mean of
        inputs (contraction, stability). No scale multipliers - scale = stimulus scale."""
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
            a = self._activate(agg - thr) + base
        return a

    # ---- forward (legacy: random projections; kept only as pure=False) ----
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
        """Clean diffusion over the real graph: ZERO random parameters.
        a = ACT(weighted_sum - thr*fan) + base. ACT = relu (legacy) or lif
        (saturating, see set_activation). Threshold = fraction of total drive
        (coincidence like a spike threshold); thr is one global protocol constant."""
        import numpy as _np
        if getattr(self, "_fan", None) is None:
            self.enable_scaling()
        a = _np.zeros(self.N, dtype=_np.float32)
        base = _np.zeros(self.N, dtype=_np.float32)
        if len(idx):
            _np.add.at(base, idx, val)
        a = base.copy()  # stimulus = current source (clamp), not initial state
        sw = (self.wM*self.sign).astype(_np.float32)  # TRAINABLE weights (not static parquet)
        CH = 2000000
        for _ in range(hops):
            agg = _np.zeros(self.N, dtype=_np.float32)
            for s in range(0, self.E, CH):
                e = slice(s, min(s+CH, self.E))
                _np.add.at(agg, self.post[e], a[self.pre[e]]*sw[e])
            agg /= (self._fan + 1e-6)
            a = self._activate(agg - thr) + base
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
        """Adaptation baseline: NEUTRAL scene (default gray 0.5), not emptiness.
        Like adaptation to background statistics - baseline must be nonzero."""
        import numpy as _np
        if image is None and self.mode == "full":
            image = _np.full((64, 64), 0.5, dtype=_np.float32)
        o = self.step(image=image)
        self._base = {k: o[k] for k in ("VIS_L", "VIS_R", "ALPN_mean", "MB_app", "MB_avo") if k in o}
        self._base_vis = None
        if self.mode == "full":
            idx, val, _ = self.encode(image=image)
            h0 = self._forward_pure(idx, val, hops=int(getattr(self, "_hops", 2)))
            self._base_vis = h0[self.VIS].astype(np.float64)
        return self._base

    def step(self, image=None, odor=None, mech=None, alpn=None, hops=None, pure=None, thr=0.0,
             odor_left=None, odor_right=None, mech_left=None, mech_right=None):
        # NOTE timescales: one step = 1 behavioral trial (seconds-minutes). The clock ticks
        # ONLY via tick_clock() - otherwise a 10Hz mob would spin a day in 2s.
        # hops=None -> self._hops (set_hops); mb default 1, full default 2.
        if hops is None:
            hops = int(getattr(self, "_hops", 2 if self.mode == "full" else 1))
        else:
            hops = int(hops)
        idx, val, info = self.encode(image=image, odor=odor, mech=mech, alpn=alpn,
                                     odor_left=odor_left, odor_right=odor_right,
                                     mech_left=mech_left, mech_right=mech_right)
        _st = getattr(self, "_std", None)
        if _st is not None and _st["alpha"] > 0:
            idx, val = self._apply_std(idx, val, _st["alpha"], _st["tau"])
        if pure is None:
            pure = True  # default: data only, zero randomness (mb and full)
        spike = (str(getattr(self, "_act", "relu")) == "spike")
        if spike and pure is False:
            raise ValueError("spike mode has no pure=False legacy; use relu/lif for that")
        if spike:
            # full LIF sim: same dict in Hz (hops/T ignored except Tms window).
            h = self._forward_spike(idx, val)
        elif pure:
            h = self._forward_pure(idx, val, hops=hops, thr=thr) if self.mode == "full" \
                else self._forward_pure_mb(idx, val, hops=hops, thr=thr)
        else:
            h = self._forward_full(idx, val, hops=hops) if self.mode == "full" else self._forward_mb(idx, val)
        kcs = h[self.KC] if (pure or spike) else h[self.KC].mean(axis=1)
        m = kcs >= np.sort(kcs)[-max(1, int(len(kcs)*0.05))]
        ks = (kcs*m).astype(np.float32)
        if spike:
            # measured MBON rates (Hz), like test_lif reports app/avo.
            r = h[self.MBON].astype(np.float32)
            kc_active = int((h[self.KC] > 0).sum())
        else:
            km_w = self.wM[self.km_mask]
            r = np.zeros(len(self.MBON)); np.add.at(r, self.km_mi, ks[self.km_ki]*km_w)
            kc_active = int(m.sum())
        pos = self._mb_pos
        ai = [pos[int(a)] for a in np.sort(self.approach)]
        vi = [pos[int(a)] for a in np.sort(self.avoid)]
        out = {
            "MB_pref": float(r[ai].mean()-r[vi].mean()),
            "MB_app": float(r[ai].mean()), "MB_avo": float(r[vi].mean()),
            "MBON": r.astype(np.float32),
            "KC_active": kc_active, "KC_overlap_n": None,
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
                out["turn_olf"] = float(out["ALPN_L"] - out["ALPN_R"])  # >0 odor on the left
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
                out["turn"] = float(out["DN_L"]-out["DN_R"])  # >0 turn left (convention)
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
                # per-neuron adaptation: mean of single-neuron dff (not dff of the mean)
                floor = 0.01*np.abs(bv).mean() + 1e-9
                for nm, m in (("VIS_L_ad", self.VIS[self.VIS_eye == 0]),
                              ("VIS_R_ad", self.VIS[self.VIS_eye == 1])):
                    b = bv[np.isin(self.VIS, m)]
                    v = h[m].astype(np.float64)
                    out[nm] = float(np.mean((v-b)/np.maximum(np.abs(b), floor)))
        if self._auto_sleep is not None:
            _, sinfo = self._sleep_tick(len(self.KC) and out["KC_active"] / max(1, len(self.KC)))
            out.update(sinfo)
        # experimental outputs (X-zone): MBON-style 2nd-hop readout, so they work
        # in mb and full alike. act = dense h with sparse KC code, then
        # X = mean over ids of sum(act[pre] * w) over cached incoming edges.
        if getattr(self, "_x_outputs", {}):
            _act = h if pure else h.mean(axis=1).astype(np.float32)
            _act = np.asarray(_act, dtype=np.float32).copy()
            try:
                _act[self.KC] = ks.astype(np.float32)
            except Exception:
                pass
            for _xn, _xids in self._x_outputs.items():
                _c = getattr(self, "_x_in", {}).get(_xn)
                if _c is None:
                    continue
                _ei, _epre, _eslot, _nn = _c
                _ei = np.asarray(_ei, dtype=np.int64)
                _w = self.wM[_ei].astype(np.float64)
                _resp = np.zeros(int(_nn), dtype=np.float64)
                np.add.at(_resp, np.asarray(_eslot, dtype=np.int64),
                          _act[np.asarray(_epre, dtype=np.int32)].astype(np.float64) * _w)
                out[f"X_{_xn}"] = float(_resp.mean())
                out[f"X_{_xn}_all"] = _resp.astype(np.float32)
        return out

    def sleep(self, episodes=1, rate=0.02):
        """Sleep (synaptic homeostasis SHY, Tononi-Cirelli): proportional downscaling of
        magnitudes toward baseline (wM0) preserving relative differences; Dale signs
        untouched (we operate on |w|); energy setpoints track the new state - no
        upward compensation (renorm in train() does not undo sleep). rate is protocol (like 0.85)."""
        for _ in range(episodes):
            self.wM[:] = (self.wM - rate * (self.wM - self.wM0)).astype(np.float32)
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        self.ref_mb[:] = cur
        if self.mode == "full":
            cur = np.zeros(self.N); np.add.at(cur, self.post, self.wM.astype(float))
            self.ref_in[:] = cur
        return {"episodes": episodes, "rate": rate}

    # ================= EXPERIMENTAL ZONE (outside pure) =================
    # x_* methods: brain growth and topology changes. REQUIRE explicit calls,
    # never fire on their own. Every change lands in the _x_log ledger;
    # x_report() tells exactly how far from v783. New-edge Dale: +1
    # (cholinergic, like KC) - a protocol assumption, logged in the ledger.
    def _x_edges(self, pa, pb, pw, ps):
        n0 = len(self.pre)
        self.pre = np.concatenate([self.pre, np.asarray(pa, np.int32)])
        self.post = np.concatenate([self.post, np.asarray(pb, np.int32)])
        self.wM = np.concatenate([self.wM, np.asarray(pw, np.float32)])
        self.sign = np.concatenate([self.sign, np.asarray(ps, np.float32)])
        self.wM0 = np.concatenate([self.wM0, np.asarray(pw, np.float32)])
        self.E = len(self.pre)
        # keep the KC->MBON mask aligned: new edges are never K->M here
        # (add_output sinks, anatomy-only growth); K->M growth rebuilds via _x_rebuild_km.
        if getattr(self, "km_mask", None) is not None and len(self.km_mask) == n0:
            self.km_mask = np.concatenate(
                [self.km_mask, np.zeros(len(self.pre) - n0, dtype=bool)])
        if getattr(self, "_fan", None) is not None and len(self._fan) < self.N:
            self._fan = np.concatenate([self._fan, np.ones(self.N - len(self._fan), np.float32)])
        if self.mode == "full":
            self.elig = np.concatenate([self.elig, np.zeros(len(self.pre) - n0, np.float32)])
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

    def x_grow_assoc(self, n=100, seed=0, wscale=0.05, per_in=6, per_out=8,
                     src_pool=None, dst_pool=None, join_kc=True):
        """GENERAL associative growth (any modality): new neurons from src_pool
        (data weight distribution x wscale) onto dst_pool. join_kc=True joins them
        to the KC pool (MB readout + DAN plasticity); False = anatomy only. Returns ids."""
        rng = np.random.default_rng(seed)
        src = self.ALPN if src_pool is None else np.asarray(src_pool, dtype=np.int32)
        dst = self.MBON if dst_pool is None else np.asarray(dst_pool, dtype=np.int32)
        isS = np.isin(self.pre, src)
        dS = self.wM[isS]
        if len(dS) == 0:
            dS = self.wM
        isD = np.isin(self.post, dst)
        dD = self.wM[isD & np.isin(self.pre, self.KC)]
        if len(dD) == 0:
            dD = self.wM[self.km_mask]
        new = np.arange(self.N, self.N + n, dtype=np.int32)
        pa, pb, pw = [], [], []
        for kk in new:
            s = rng.choice(src, size=min(per_in, len(src)), replace=False)
            pa.extend(s); pb.extend([kk] * len(s))
            pw.extend(rng.choice(dS, size=len(s)) * wscale)
            dd = rng.choice(dst, size=min(per_out, len(dst)), replace=False)
            pa.extend([kk] * len(dd)); pb.extend(dd)
            pw.extend(rng.choice(dD, size=len(dd)) * wscale)
        self.N += n
        if join_kc:
            self.KC = np.sort(np.concatenate([self.KC, new]))
        n0 = self._x_edges(pa, pb, pw, np.ones(len(pw), np.float32))
        if join_kc:
            self._x_rebuild_km()
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "grow_assoc", "n": int(n), "edges": len(pw), "wscale": wscale,
                            "join_kc": bool(join_kc), "dale": "+1(cholinergic)", "i0": int(n0)})
        return new

    def x_grow_kc(self, n=100, seed=0, wscale=0.05, per_kc_in=6, per_kc_out=8, src_pool=None):
        """Olfactory pseudo-KC = x_grow_assoc(src=ALPN/bias, dst=MBON, join_kc). Back-compat wrapper."""
        return self.x_grow_assoc(n=n, seed=seed, wscale=wscale, per_in=per_kc_in,
                                 per_out=per_kc_out, src_pool=src_pool, dst_pool=None, join_kc=True)

    def x_add_edge(self, a, b, w, dale=+1.0):
        """Single edge (ledgered)."""
        n0 = self._x_edges([a], [b], [abs(w)], [dale])
        self._x_rebuild_km()
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "add_edge", "a": int(a), "b": int(b), "w": float(w),
                            "dale": float(dale), "i0": int(n0)})
        return n0

    def x_cut_edge(self, i):
        """Functional cut (weight->0, reversible entry; topology in the ledger)."""
        i = int(i)
        old = float(self.wM[i])
        self.wM[i] = 0.0
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "cut_edge", "i": i, "a": int(self.pre[i]), "b": int(self.post[i]),
                            "w_old": old})
        return old

    def x_prune_kc(self, ids, odors=("ethyl_hexanoate", "geosmin")):
        """Developmental pruning: new KC absent from every probe's top-k get their
        outputs (K->M) zeroed. One ledger entry. Returns (pruned, active)."""
        ids = np.asarray(ids, dtype=np.int32)
        act = np.zeros(len(ids), bool)
        _hp = int(getattr(self, "_hops", 2))
        for od in odors:
            h = self._forward_pure(*self.encode(odor=od)[:2], hops=_hp, thr=0.0)
            kcs = h[self.KC]; k = max(1, int(len(kcs) * 0.05))
            top = set(np.argsort(kcs)[-k:])
            kpos = {int(g): i for i, g in enumerate(np.sort(self.KC))}
            act |= np.array([kpos[int(x)] in top for x in ids])
        dead = ids[~act]
        if len(dead):
            m = np.isin(self.pre, dead) & self.km_mask
            self.wM[m] = 0.0
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "prune_kc", "dead": int(len(dead)), "of": int(len(ids))})
        return int(len(dead)), int(act.sum())

    def x_code(self, odor):
        """Odor's top-k KC code (position set). Dictionary registry in _x_codes."""
        h = self._forward_pure(*self.encode(odor=odor)[:2],
                               hops=int(getattr(self, "_hops", 2)), thr=0.0)
        kcs = h[self.KC]; k = max(1, int(len(kcs) * 0.05))
        return set(np.argsort(kcs)[-k:].tolist())

    def x_novelty(self, odor):
        """1 - max code overlap with remembered ones (0=known, 1=novel)."""
        reg = getattr(self, "_x_codes", {})
        if not reg:
            return 1.0
        code = self.x_code(odor)
        return 1.0 - max(len(code & v) / max(len(code), 1) for v in reg.values())

    def x_remember(self, odor):
        self._x_codes = getattr(self, "_x_codes", {})
        self._x_codes[odor] = self.x_code(odor)
        return len(self._x_codes[odor])

    def x_neurogenesis(self, odor, n=500, seed=0, wscale=0.01, per_kc_in=4, ov_thr=0.05):
        """Interference-gated neurogenesis: grows IFF the odor's code collides
        (>ov_thr) with an OPPOSITE-valence memory (overwrite risk). Newborns are
        SELECTIVE: inputs from ALPN active for THIS odor (drive pool, not random).
        Returns dict(grew, overlap, novelty). Biology-style: novelty + utility."""
        code = self.x_code(odor)
        reg = getattr(self, "_x_codes", {})
        ov = 0.0
        for o, v in reg.items():
            if o != odor:
                ov = max(ov, len(code & v) / max(len(code), 1))
        nov = 1.0 - max([len(code & v) / max(len(code), 1) for v in reg.values()] or [1.0])
        if ov < ov_thr:
            return {"grew": False, "overlap": round(ov, 3), "novelty": round(nov, 3)}
        # ALPN pool active for this odor (DoOR profile - birth selectivity)
        import os as _os
        dd = np.load(f"{self.path}/door_odors.npz")
        oi, ovv = dd[odor + "_idx"], np.abs(dd[odor + "_val"])
        # ALPN: top by 1-hop strength from alpn_patterns
        pat = np.load(f"{self.path}/alpn_patterns.npz")
        sc = np.abs(pat[odor]); AG = pat["ALPN_glob"]
        pool = np.array(AG[np.argsort(sc)[-100:]], dtype=np.int32)
        new = self.x_grow_kc(n=n, seed=seed, wscale=wscale, per_kc_in=per_kc_in, src_pool=pool)
        self._x_log.append({"op": "neurogenesis", "odor": odor, "n": int(n), "overlap": round(ov, 3)})
        return {"grew": True, "overlap": round(ov, 3), "novelty": round(nov, 3), "n": int(n)}

    def x_report(self):
        """How much added/cut + assumptions. Returns dict, prints."""
        lg = getattr(self, "_x_log", [])
        added = sum(e.get("edges", 1) for e in lg if e["op"] in ("grow_kc", "grow_assoc", "add_edge", "add_output", "teach_output"))
        cut = sum(1 for e in lg if e["op"] == "cut_edge")
        grown = sum(e.get("n", 0) for e in lg if e["op"] in ("grow_kc", "grow_assoc", "add_output"))
        outs = {k: len(v) for k, v in getattr(self, "_x_outputs", {}).items()}
        try:
            _ei = float((self.wM * self.sign).sum())
            _e0 = float((self.wM0 * self.sign).sum())
            _drift = abs(_ei - _e0) / max(1e-9, abs(_e0)) * 100.0
        except Exception:
            _drift = 0.0
        rep = {"new_neurons": grown, "added_edges": added, "cut_edges": cut,
                "N": self.N, "E": getattr(self, "E", len(self.pre)),
                "ops": len(lg), "dale_new": "+1(cholinergic)", "x_outputs": outs,
                "EI_drift_%": round(_drift, 4)}
        print(f"X-REPORT: +{grown} neurons, +{added} edges, ~{cut} cuts, N={self.N} E={getattr(self, 'E', len(self.pre))} outputs={outs} drift={_drift:.4f}%", flush=True)
        return rep

    def x_add_output(self, name, n=8, src_pool=None, per_in=None, wscale=None, seed=0):
        """Arbitrary new output (X-zone sink readout).

        Creates n new neurons driven by src_pool (default KC - the associative
        code, so the output binds stimulus conjunctions, i.e. circumstances).
        Pass any pool for other circumstances, e.g. np.concatenate([KC, ME])
        for odor+vision, ALPN for innate-like, EFFERENT-adjacent for motor-like.
        Readout appears in step() as X_<name> (mean) + X_<name>_all (vector).
        Teaching via x_teach_output. Ledgered. Returns ids.

        Defaults are mode-aware (full signals are fan-attenuated): mb keeps
        per_in=8/wscale=0.05, full uses per_in=64/wscale=0.5. Pass explicit
        values to override."""
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise ValueError("output name must be a non-empty string")
        outs = getattr(self, "_x_outputs", {})
        if name in outs:
            raise ValueError(f"output '{name}' exists; pick another name")
        rng = np.random.default_rng(seed)
        src = self.KC if src_pool is None else np.asarray(src_pool, dtype=np.int32)
        if len(src) == 0:
            raise ValueError("src_pool is empty")
        if per_in is None:
            per_in = 8 if self.mode == "mb" else 64
        if wscale is None:
            wscale = 0.05 if self.mode == "mb" else 0.5
        isS = np.isin(self.pre, src)
        dS = self.wM[isS] if int(isS.sum()) else self.wM
        new = np.arange(self.N, self.N + n, dtype=np.int32)
        pa, pb, pw = [], [], []
        for kk in new:
            s = rng.choice(src, size=min(per_in, len(src)), replace=False)
            pa.extend(s.tolist())
            pb.extend([int(kk)] * len(s))
            pw.extend((rng.choice(dS, size=len(s)) * wscale).tolist())
        self.N += n
        n0 = self._x_edges(pa, pb, pw, np.ones(len(pw), np.float32))
        if not hasattr(self, "E"):
            self.E = len(self.pre)
        outs[name] = np.asarray(new, dtype=np.int32)
        self._x_outputs = outs
        # cache incoming edges for the MBON-style readout (edge idx, pre, slot).
        _slot = {int(v): i for i, v in enumerate(new)}
        _ei = np.arange(int(n0), int(n0) + len(pa), dtype=np.int64)
        _epre = np.asarray(pa, dtype=np.int32)
        _eslot = np.asarray([_slot[int(b)] for b in pb], dtype=np.int64)
        self._x_in = getattr(self, "_x_in", {})
        self._x_in[name] = (_ei, _epre, _eslot, int(n))
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "add_output", "name": name, "n": int(n),
                            "edges": len(pw), "wscale": float(wscale),
                            "per_in": int(per_in), "dale": "+1(cholinergic)",
                            "i0": int(n0)})
        return np.asarray(new, dtype=np.int32)

    def x_list_outputs(self):
        """Registered X-zone outputs: {name: n_neurons}."""
        return {k: int(len(v)) for k, v in getattr(self, "_x_outputs", {}).items()}

    def x_teach_output(self, name, trials=3, eta=0.3, high=True, target=None,
                       pure=True, thr=0.0, hops=None, scope="new", eta_full=0.02,
                       **stim):
        """Teach an X output when/in which circumstances to fire.

        stim = any step() stimulus (odor, image, mech, alpn, odor_left/right,
        mech_left/right, dan_rew/dan_pun) - i.e. the circumstance, freely
        combined (odor + shape + touch at once). Two modes:

        - binary (target=None): high=True for GO examples (fire here),
          high=False for NOGO (stay silent); alternate for discrimination.
        - value (target=float): regression to an exact readout value in this
          circumstance, e.g. target=1.0 for (geosmin + triangle), 0.0 for
          (ethyl + square), 0.5 for a third mix. Alternate the circumstances
          round-robin; the KC code separates conjunctions, so one output can
          hold several odor+shape+touch -> value pairs.

        scope (X-zone, ledgered):
        - scope='new' (default, safe): only the output's own incoming edges
          change; the frozen brain is untouched (drift 0%).
        - scope='whole' (experimental): the whole brain learns too - every
          edge gets a small Hebbian nudge dw = eta_full * err * coincidence
          (+/-2% clip per edge, signs frozen). This can rewire upstream
          paths (e.g. pull vision deeper toward KC) but drifts identity;
          check drift via x_report()['EI_drift_%'] and wash with sleep().
          mb learns with eta=0.3-0.5 / eta_full~0.05; full needs
          eta~1.0-1.5 / eta_full~0.01-0.02.

        Rule (local, frozen signs kept, |w| clip [0.05, 650]):
        binary: w *= 1 +/- eta * pre_norm; value: normalized delta
        w += eta * (target - cur) * pre_norm / (active_per_neuron * max_pre).
        Practical doses: mb 2-4 rounds per circumstance; full ~5-10 rounds.
        hops=None -> default from set_hops()."""
        if hops is None:
            hops = int(getattr(self, "_hops", 2 if self.mode == "full" else 1))
        else:
            hops = int(hops)
        outs = getattr(self, "_x_outputs", {})
        if name not in outs:
            raise ValueError(f"unknown output '{name}'; x_add_output() first")
        _c = getattr(self, "_x_in", {}).get(name)
        if _c is None:
            raise ValueError(f"output '{name}' has no incoming edges")
        _ei, _epre, _eslot, _nn = _c
        _ei = np.asarray(_ei, dtype=np.int64)
        _epre = np.asarray(_epre, dtype=np.int32)
        hist = []
        _spk = (str(getattr(self, "_act", "relu")) == "spike")
        for _ in range(max(1, int(trials))):
            idx, val, _ = self.encode(**stim)
            if _spk:
                # Hebbian update on measured rates (slow; prefer rate modes).
                h = self._forward_spike(idx, val)
                a = h.astype(np.float64)
            elif pure:
                h = self._forward_pure(idx, val, hops=hops, thr=thr) if self.mode == "full" \
                    else self._forward_pure_mb(idx, val, hops=hops, thr=thr)
                a = h.astype(np.float64)
            else:
                h = self._forward_full(idx, val, hops=hops) if self.mode == "full" else self._forward_mb(idx, val)
                a = h.mean(axis=1).astype(np.float64)
            # sparse KC code drives the update (same act as the readout).
            try:
                _k = a[self.KC]
                _m = _k >= np.sort(_k)[-max(1, int(len(_k) * 0.05))]
                a[self.KC] = (_k * _m)
            except Exception:
                pass
            pre_act = a[_epre]
            mx = float(pre_act.max()) if len(pre_act) else 0.0
            pn = (pre_act / mx).astype(np.float32) if mx > 1e-9 else np.zeros_like(pre_act, dtype=np.float32)
            if target is not None:
                tgt = float(target)
                wcur = self.wM[_ei].astype(np.float64)
                _resp = np.zeros(int(_nn), dtype=np.float64)
                np.add.at(_resp, np.asarray(_eslot, dtype=np.int64),
                          a[_epre].astype(np.float64) * wcur)
                cur = float(_resp.mean())
                err = tgt - cur
            elif high:
                err = 1.0
                cur = 0.0
            else:
                err = -1.0
                cur = 0.0
            if target is not None:
                _slot = np.asarray(_eslot, dtype=np.int64)
                _cnt = np.bincount(_slot, weights=(pn > 1e-6).astype(np.float64),
                                   minlength=int(_nn))
                _den = _cnt[_slot] * mx
                _step = np.zeros_like(pn, dtype=np.float64)
                _ok = _den > 1e-12
                _step[_ok] = float(eta) * err * pn[_ok].astype(np.float64) / _den[_ok]
                self.wM[_ei] = np.clip(
                    wcur + _step, 0.05, 650.0).astype(np.float32)
            elif high:
                self.wM[_ei] = np.clip(
                    self.wM[_ei] * (1.0 + float(eta) * pn), 0.05, 650.0).astype(np.float32)
            else:
                self.wM[_ei] = np.clip(
                    self.wM[_ei] * (1.0 - float(eta) * pn), 0.05, 650.0).astype(np.float32)
            if scope == "whole":
                # experimental whole-brain nudge: every edge moves a little
                # along its coincidence, direction set by the output error.
                # Chunked (15M edges), signs frozen, +/-2% per edge.
                _mx = float(mx) if mx > 1e-12 else 1.0
                _ef = float(eta_full) * float(err)
                if _ef != 0.0:
                    for _s in range(0, len(self.pre), self.CH):
                        _e = slice(_s, min(_s + self.CH, len(self.pre)))
                        _c = (a[self.pre[_e]].astype(np.float64) *
                              a[self.post[_e]].astype(np.float64)) / (_mx * _mx)
                        _dw = np.clip(_ef * _c, -0.02, 0.02)
                        self.wM[_e] = np.clip(
                            self.wM[_e].astype(np.float64) * (1.0 + _dw),
                            0.05, 650.0).astype(np.float32)
            o = self.step(pure=pure, thr=thr, hops=hops, **stim)
            hist.append(float(o.get(f"X_{name}", 0.0)))
        self._x_log = getattr(self, "_x_log", [])
        self._x_log.append({"op": "teach_output", "name": name, "trials": int(trials),
                            "eta": float(eta), "high": bool(high),
                            "target": None if target is None else float(target),
                            "scope": scope, "eta_full": float(eta_full),
                            "stim": sorted(stim.keys())})
        return {"name": name, "high": bool(high), "responses": hist,
                "final": hist[-1] if hist else 0.0}

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
              odor_left=None, odor_right=None, mech_left=None, mech_right=None, gated=True, hops=None):
        """reward>0 (PAM: weakens avoid) / punish>0 (PPL1: weakens approach). Returns post-learning step().
        gated=True: memory-protecting DAN - depression weighted by KC uniqueness against
        registered codes (x_remember): shared KC spared (x1.0), unique x0.85.
        No codes: legacy slab. gated=False: always legacy slab.
        hops=None -> default from set_hops() (full 2, mb 1)."""
        if hops is None:
            hops = int(getattr(self, "_hops", 2 if self.mode == "full" else 1))
        else:
            hops = int(hops)
        idx, val, _ = self.encode(image=image, odor=odor, mech=mech, alpn=alpn,
                                  odor_left=odor_left, odor_right=odor_right,
                                  mech_left=mech_left, mech_right=mech_right)
        if str(getattr(self, "_act", "relu")) == "spike":
            h = self._forward_spike(idx, val)
            kcs = h[self.KC]
        elif pure:
            h = self._forward_pure(idx, val, hops=hops, thr=thr) if self.mode == "full" \
                else self._forward_pure_mb(idx, val, hops=hops, thr=thr)
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
        cur_od = odor if isinstance(odor, str) else None
        reg = getattr(self, "_x_codes", {})
        if gated and reg and cur_od is not None:
            # per-KC depression weight: 1.0 for code-shared -> 0.85 unique
            others = [v for o, v in reg.items() if o != cur_od]
            kmki = self.km_ki
            if others:
                shared = np.zeros(len(kmki), dtype=np.float32)
                for v in others:
                    shared += np.isin(kmki, list(v)).astype(np.float32)
                fkc = 0.85 + 0.15 * np.minimum(1.0, shared)
            else:
                fkc = np.full(len(kmki), 0.85, np.float32)
        else:
            fkc = None
        if reward > 0:
            sel = self.km_is_avoid & m[self.km_ki]
            if fkc is None:
                self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*0.85, self.wM[self.km_mask])
            else:
                self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*fkc, self.wM[self.km_mask])
        if punish > 0:
            sel = self.km_is_approach & m[self.km_ki]
            if fkc is None:
                self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*0.85, self.wM[self.km_mask])
            else:
                self.wM[self.km_mask] = np.where(sel, self.wM[self.km_mask]*fkc, self.wM[self.km_mask])
        self.wM[self.km_mask] = np.maximum(self.wM[self.km_mask], 0.05)
        cur = np.zeros(len(self.MBON)); np.add.at(cur, self.km_mi, self.wM[self.km_mask].astype(float))
        sc = np.ones(len(self.MBON)); nz = cur > 1e-9
        sc[nz] = np.clip(self.ref_mb[nz]/cur[nz], 0.9, 1.1)
        self.wM[self.km_mask] = (self.wM[self.km_mask]*sc[self.km_mi]).astype(np.float32)
        return self.step(image=image, odor=odor, mech=mech, alpn=alpn,
                         odor_left=odor_left, odor_right=odor_right,
                         mech_left=mech_left, mech_right=mech_right, hops=hops)
