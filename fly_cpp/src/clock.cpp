// clock.cpp — Goldbeter 1995 PER-TIM TTFL 1:1 with clock.py.
// Units uM/h, 1 step = 1h, dt=0.05, light degrades TIM via Cry.
#include "fly.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fly {

void FlyBrain::enable_clock(float amp) {
    clock_amp = amp;
    has_clock = true;
    ck_M = 0.2; ck_P0 = 0.3; ck_P1 = 0.3; ck_P2 = 1.5; ck_PN = 0.5;
    ck_t = 0.0; ck_avg = -1.0;
}

std::map<std::string, double> FlyBrain::tick_clock(double hours, float light) {
    if (!has_clock) throw std::runtime_error("enable_clock() first");
    float li = (light < 0) ? ambient : light;
    const double vs = 0.76, vm = 0.65, Km = 0.5, KI = 1.0;
    const int nn = 4;
    const double ks = 0.38;
    const double V1 = 3.2, V2 = 1.58, V3 = 5.0, V4 = 2.5;
    const double K1 = 2.0, K2 = 2.0, K3 = 2.0, K4 = 2.0;
    const double k1 = 1.9, k2 = 1.3, vd0 = 0.95, Kd = 0.2;
    const double dt = 0.05, vd_light = 1.5;
    int n = std::max(1, (int)std::llround(hours));
    for (int h = 0; h < n; h++) {
        double vd = vd0 * (1.0 + vd_light * li);
        for (int i = 0; i < (int)(1.0 / dt); i++) {
            double M = ck_M, P0 = ck_P0, P1 = ck_P1, P2 = ck_P2, PN = ck_PN;
            double KIn = std::pow(KI, nn);
            double dM = vs * KIn / (KIn + std::pow(PN, nn)) - vm * M / (Km + M);
            double dP0 = ks * M - V1 * P0 / (K1 + P0) + V2 * P1 / (K2 + P1);
            double dP1 = V1 * P0 / (K1 + P0) - V2 * P1 / (K2 + P1) -
                         V3 * P1 / (K3 + P1) + V4 * P2 / (K4 + P2);
            double dP2 = V3 * P1 / (K3 + P1) - V4 * P2 / (K4 + P2) -
                         vd * P2 / (Kd + P2) - k1 * P2 + k2 * PN;
            double dPN = k1 * P2 - k2 * PN;
            ck_M = std::max(0.0, M + dM * dt);
            ck_P0 = std::max(0.0, P0 + dP0 * dt);
            ck_P1 = std::max(0.0, P1 + dP1 * dt);
            ck_P2 = std::max(0.0, P2 + dP2 * dt);
            ck_PN = std::max(0.0, PN + dPN * dt);
        }
        ck_t += 1.0;
    }
    if (ck_avg < 0) ck_avg = ck_P2;
    else ck_avg += (ck_P2 - ck_avg) / 24.0;
    const double span = 0.8;
    ck_morning = (float)std::min(1.0, std::max(0.0, 0.5 + (ck_avg - ck_P2) / span));
    ck_night = ck_P2 > ck_avg;
    ck_night_frac = (float)(ck_P2 / std::max(ck_P2 + ck_PN + 0.5, 1e-9));
    return {{"t", ck_t}, {"M", ck_M}, {"P2", ck_P2}, {"PN", ck_PN},
            {"morning", ck_morning}, {"night_frac", ck_night_frac},
            {"night", ck_night ? 1.0 : 0.0}};
}

} // namespace fly
