"""Artificial molecular-clock TTFL (Goldbeter 1995, Drosophila PER-TIM).
Rationale: TTFL is intracellular biochemistry - absent from the connectome by definition
(maps synapses, not proteins), like phototransduction. The box replaces a missing organ,
it does not compute perception. Signal goes to the CORRECT neurons (s-LNv=morning, LNd/DN1=evening).
Light (environment) degrades TIM via Cry - entrainment like in biology.
Time unit: 1 step = 1h (day=24 steps); ck_dt for ODE accuracy.
"""
import numpy as np


class TTFL:
    def __init__(self, light=0.0):
        # Goldbeter 1995 Proc R Soc B - pelne rownania, jednostki uM/h, okres ~24h
        self.vs, self.vm, self.Km, self.KI, self.n = 0.76, 0.65, 0.5, 1.0, 4
        self.ks = 0.38
        self.V1, self.V2, self.V3, self.V4 = 3.2, 1.58, 5.0, 2.5
        self.K1, self.K2, self.K3, self.K4 = 2.0, 2.0, 2.0, 2.0
        self.k1, self.k2, self.vd, self.Kd = 1.9, 1.3, 0.95, 0.2
        self.M, self.P0, self.P1, self.P2, self.PN = 0.2, 0.3, 0.3, 1.5, 0.5
        self.light = light
        self.t = 0.0
        self._avg = None  # kroczaca srednia P2 (odniesienie fazy, tau~24h)

    def step(self, light, dt=0.05, vd_light=1.5):
        """One step=1h. light 0..1 (environment) -> TIM degradation (Cry)."""
        vd = self.vd * (1.0 + vd_light * light)  # Cry: light degrades TIM
        for _ in range(int(1.0 / dt)):
            M, P0, P1, P2, PN = self.M, self.P0, self.P1, self.P2, self.PN
            dM = self.vs * self.KI ** self.n / (self.KI ** self.n + PN ** self.n) \
                - self.vm * M / (self.Km + M)
            dP0 = self.ks * M - self.V1 * P0 / (self.K1 + P0) + self.V2 * P1 / (self.K2 + P1)
            dP1 = self.V1 * P0 / (self.K1 + P0) - self.V2 * P1 / (self.K2 + P1) \
                - self.V3 * P1 / (self.K3 + P1) + self.V4 * P2 / (self.K4 + P2)
            dP2 = self.V3 * P1 / (self.K3 + P1) - self.V4 * P2 / (self.K4 + P2) \
                - vd * P2 / (self.Kd + P2) - self.k1 * P2 + self.k2 * PN
            dPN = self.k1 * P2 - self.k2 * PN
            self.M += dM * dt; self.P0 += dP0 * dt; self.P1 += dP1 * dt
            self.P2 += dP2 * dt; self.PN += dPN * dt
            for k in ("M", "P0", "P1", "P2", "PN"):
                if getattr(self, k) < 0:
                    setattr(self, k, 0.0)
        self.t += 1.0
        # phase vs trailing mean: high TIM = biological night
        self._avg = self.P2 if self._avg is None else self._avg + (self.P2 - self._avg) / 24.0
        span = 0.8  # typowa amplituda P2 w DD (0.18-1.00)
        morning = float(np.clip(0.5 + (self._avg - self.P2) / span, 0.0, 1.0))
        night = bool(self.P2 > self._avg)
        return {"t": self.t, "M": self.M, "P2": self.P2, "PN": self.PN,
                "morning": morning, "night_frac": float(self.P2 / max(self.P2 + self.PN + 0.5, 1e-9)),
                "night": night}


if __name__ == "__main__":
    ck = TTFL()
    pks, pns = [], []
    for h in range(240):
        s = ck.step(0.0)
        pks.append(s["P2"]); pns.append(s["PN"])
    pks = np.array(pks)
    # okres: autokorelacja P2
    ac = np.correlate(pks - pks.mean(), pks - pks.mean(), "full")[len(pks):]
    ac[0] = 0
    per = int(np.argmax(ac[:72])) if ac[:72].max() > 0 else -1
    print(f"DD period={per}h (target 24), P2 range={pks.min():.2f}-{pks.max():.2f}")
    # entrainment LD 12:12
    ck2 = TTFL()
    ph = []
    for h in range(120):
        s = ck2.step(1.0 if h % 24 < 12 else 0.0)
        ph.append(s["P2"])
    ph = np.array(ph)
    print(f"LD: stable phase (last 2 days corr)={np.corrcoef(ph[-48:-24], ph[-24:])[0,1]:.3f}")
