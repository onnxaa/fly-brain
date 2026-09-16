// spike.cpp — LIF spiking forward 1:1 with fly_api._forward_spike
// (test_lif v2 equations). Poisson drive, g-reset, 2-step refractory and
// delay, signed weights, APL x4, graded early vision in full mode.
// RNG note: PCG64 (NumPy) vs mt19937_64 here — rates agree within
// sampling noise, not bit-wise.
#include "fly.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <vector>

namespace fly {

std::vector<float> FlyBrain::forward_spike(const std::vector<int32_t>& idx,
                                           const std::vector<float>& val,
                                           bool plastic) {
    const int T = spike_T;
    const double V0 = -52.0, VRST = -52.0, VTH = -45.0, DT = 1.0;
    const double DEC_G = std::exp(-DT / 5.0), K_MBR = DT / 20.0;
    const double W_SYN = 0.275, APL_F = 4.0, KREL = 20.0;
    const double WDRV = spike_wdrv;
    // extended graph (mb APL nodes) or base graph
    int nN = N;
    std::vector<int32_t> pX, qX;
    std::vector<double> wX;
    std::vector<int32_t> apl_idx;
    std::vector<char> graded;
    if (mode == "mb" && !sp_k2a_pre.empty()) {
        int A0 = N;
        nN = N + 2;
        pX = pre; qX = post;
        wX.resize(E);
        for (int64_t e = 0; e < E; e++) wX[(size_t)e] = (double)wM[(size_t)e] * sign[(size_t)e];
        for (size_t i = 0; i < sp_k2a_pre.size(); i++) {
            pX.push_back(sp_k2a_pre[i]); qX.push_back(A0 + sp_k2a_apl[i]);
            wX.push_back(sp_k2a_w[i]);
        }
        for (size_t i = 0; i < sp_a2k_post.size(); i++) {
            pX.push_back(A0 + sp_a2k_apl[i]); qX.push_back(sp_a2k_post[i]);
            wX.push_back(-(double)sp_a2k_w[i]);
        }
        apl_idx = {A0, A0 + 1};
        graded.assign(nN, 0);
    } else {
        pX = pre; qX = post;
        wX.resize(E);
        for (int64_t e = 0; e < E; e++) wX[(size_t)e] = (double)wM[(size_t)e] * sign[(size_t)e];
        apl_idx = spike_apl;
        graded.assign(nN, 0);
        for (auto g : spike_gset)
            if (g >= 0 && g < nN) graded[(size_t)g] = 1;
    }
    int64_t EX = (int64_t)pX.size();
    // CSR by pre for event propagation
    std::vector<int64_t> hpre(nN + 1, 0);
    for (int64_t e = 0; e < EX; e++) hpre[pX[(size_t)e] + 1]++;
    for (int i = 0; i < nN; i++) hpre[i + 1] += hpre[i];
    std::vector<int32_t> cpost(EX);
    std::vector<double> cw(EX);
    std::vector<int64_t> cur = hpre;
    std::vector<int32_t> cidx((size_t)EX);
    for (int64_t e = 0; e < EX; e++) {
        int64_t s = cur[pX[(size_t)e]]++;
        cpost[(size_t)s] = qX[(size_t)e];
        cw[(size_t)s] = wX[(size_t)e];
        cidx[(size_t)s] = (int32_t)e;
    }
    // trace-STDP structures (plastic only): CSR by post over slots + traces.
    // wX index < E maps to wM (base+X edges); mb APL-extension slots skipped.
    std::vector<int64_t> hpostB;
    std::vector<int32_t> bslot;
    std::vector<double> tr_pre, tr_post;
    const double DEC_T = std::exp(-DT / stdp_tau);
    if (plastic) {
        hpostB.assign((size_t)nN + 1, 0);
        for (int64_t s = 0; s < EX; s++) hpostB[(size_t)cpost[(size_t)s] + 1]++;
        for (int i = 0; i < nN; i++) hpostB[(size_t)i + 1] += hpostB[(size_t)i];
        bslot.assign((size_t)EX, 0);
        std::vector<int64_t> cur2(hpostB.begin(), hpostB.begin() + nN);
        for (int64_t s = 0; s < EX; s++) {
            int64_t t = cur2[(size_t)cpost[(size_t)s]]++;
            bslot[(size_t)t] = (int32_t)s;
        }
        tr_pre.assign((size_t)nN, 0.0);
        tr_post.assign((size_t)nN, 0.0);
    }
    bool carry = (state_leak > 0 && spk_nN == nN &&
                  (int)spk_v.size() == nN);
    std::vector<double> v(nN, V0), vth(nN, VTH), gg(nN, 0.0), adapt(nN, 0.0);
    for (auto a : apl_idx)
        if (a >= 0 && a < nN) vth[(size_t)a] = V0 + APL_F * (VTH - V0);
    std::vector<int> refr(nN, 0);
    if (carry) {
        // exact continuation: deviations decay by leak (= inter-trial gap)
        for (int i = 0; i < nN; i++) {
            v[(size_t)i] = V0 + (spk_v[(size_t)i] - V0) * state_leak;
            gg[(size_t)i] = spk_gg[(size_t)i] * state_leak;
            adapt[(size_t)i] = spk_adapt[(size_t)i] * state_leak;
            refr[(size_t)i] = spk_refr[(size_t)i];
        }
        spk_calls++;
    }
    std::vector<int> nsp(nN, 0), nrel(nN, 0);
    std::vector<int32_t> didx;
    std::vector<double> dval;
    for (size_t i = 0; i < idx.size(); i++)
        if (idx[i] < nN) {
            didx.push_back(idx[i]);
            dval.push_back(i < val.size() ? val[i] : 0.0f);
        }
    std::vector<double> dp(didx.size());
    for (size_t i = 0; i < didx.size(); i++)
        dp[i] = dval[i] * spike_rmax / 2.0 * DT / 1000.0;
    std::vector<int> G;
    for (int i = 0; i < nN; i++)
        if (graded[(size_t)i]) G.push_back(i);
    std::deque<std::vector<double>> dq;
    dq.emplace_back(nN, 0.0);
    dq.emplace_back(nN, 0.0);
    if (carry)
        for (int i = 0; i < nN; i++) {
            dq[0][(size_t)i] = spk_dq0[(size_t)i] * state_leak;
            dq[1][(size_t)i] = spk_dq1[(size_t)i] * state_leak;
        }
    std::mt19937_64 rng(carry ? (uint64_t)spike_seed + spk_calls
                              : (uint64_t)spike_seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double DEC_A = std::exp(-DT / 100.0), A_INC = spike_ainc;
    const double AD_TAU = 50.0, AD_FLOOR = spike_adapt;
    const int BURN = spike_burn;
    std::vector<double> tmp(nN);
    std::vector<int> fire_idx;
    fire_idx.reserve(4096);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < nN; i++) gg[(size_t)i] *= DEC_G;
        for (int i = 0; i < nN; i++) adapt[(size_t)i] *= DEC_A;
        std::vector<double>& front = dq.front();
        for (int i = 0; i < nN; i++) gg[(size_t)i] += front[(size_t)i];
        dq.pop_front();
        dq.emplace_back(nN, 0.0);
        if (!didx.empty()) {
            double ad = AD_FLOOR + (1.0 - AD_FLOOR) * std::exp(-t / AD_TAU);
            fire_idx.clear();
            for (size_t i = 0; i < didx.size(); i++)
                if (U(rng) < std::fabs(dp[i]) * ad) fire_idx.push_back((int)i);
            for (int fi : fire_idx) {
                if (dp[(size_t)fi] > 0) gg[(size_t)didx[(size_t)fi]] += WDRV;
                else gg[(size_t)didx[(size_t)fi]] -= WDRV;
            }
        }
        for (int i = 0; i < nN; i++) {
            if (refr[(size_t)i] == 0 && !graded[(size_t)i])
                v[(size_t)i] += K_MBR * (V0 - v[(size_t)i] + gg[(size_t)i] - adapt[(size_t)i]);
            else if (graded[(size_t)i])
                v[(size_t)i] += K_MBR * (V0 - v[(size_t)i] + gg[(size_t)i]);
        }
        for (int i = 0; i < nN; i++)
            if (refr[(size_t)i] > 0) refr[(size_t)i]--;
        std::vector<int> sp, rel, ev;
        for (int i = 0; i < nN; i++)
            if (refr[(size_t)i] == 0 && !graded[(size_t)i] && v[(size_t)i] >= vth[(size_t)i])
                sp.push_back(i);
        for (int g : G)
            if (U(rng) < std::max(0.0, v[(size_t)g] - V0) * KREL * DT / 1000.0) rel.push_back(g);
        ev = sp;
        ev.insert(ev.end(), rel.begin(), rel.end());
        if (t < BURN) {
            for (int s : sp) {
                v[(size_t)s] = VRST; gg[(size_t)s] = 0.0;
                adapt[(size_t)s] += A_INC; refr[(size_t)s] = 2;
            }
            if (!ev.empty()) {
                std::vector<double>& back = dq.back();
                for (int s : ev)
                    for (int64_t e = hpre[s]; e < hpre[s + 1]; e++)
                        back[(size_t)cpost[(size_t)e]] += cw[(size_t)e] * W_SYN;
            }
            continue;
        }
        for (int s : sp) {
            v[(size_t)s] = VRST; gg[(size_t)s] = 0.0;
            adapt[(size_t)s] += A_INC; refr[(size_t)s] = 2;
            nsp[(size_t)s]++;
        }
        if (plastic && t >= BURN) {
            for (size_t i = 0; i < tr_pre.size(); i++) {
                tr_pre[i] *= DEC_T; tr_post[i] *= DEC_T;
            }
        }
        if (plastic && t >= BURN && !sp.empty()) {
            // trace-STDP on wM magnitudes (true spikes only, burn excluded).
            // Valence stays in the DAN slab in train() (no third factor yet).
            #pragma omp parallel for schedule(static) if(sp.size() > 64)
            for (size_t si = 0; si < sp.size(); si++) {
                int s = sp[si];
                for (int64_t e = hpre[s]; e < hpre[s + 1]; e++) {
                    int64_t o = cidx[(size_t)e];
                    if (o < 0 || o >= E) continue;
                    int32_t q = cpost[(size_t)e];
                    double nw = (double)wM[(size_t)o] - stdp_Aminus * tr_post[(size_t)q];
                    if (nw < 0.05) nw = 0.05; if (nw > 650.0) nw = 650.0;
                    wM[(size_t)o] = (float)nw;
                    cw[(size_t)e] = nw * sign[(size_t)o];
                }
            }
            #pragma omp parallel for schedule(static) if(sp.size() > 64)
            for (size_t si = 0; si < sp.size(); si++) {
                int q = sp[si];
                for (int64_t k = hpostB[q]; k < hpostB[q + 1]; k++) {
                    int64_t e = bslot[(size_t)k];
                    int64_t o = cidx[(size_t)e];
                    if (o < 0 || o >= E) continue;
                    int32_t pp = (int32_t)pX[(size_t)o];
                    double nw = (double)wM[(size_t)o] + stdp_Aplus * tr_pre[(size_t)pp];
                    if (nw < 0.05) nw = 0.05; if (nw > 650.0) nw = 650.0;
                    wM[(size_t)o] = (float)nw;
                    cw[(size_t)e] = nw * sign[(size_t)o];
                }
            }
            for (int s : sp) { tr_pre[(size_t)s] += 1.0; tr_post[(size_t)s] += 1.0; }
        }
        for (int s : rel) nrel[(size_t)s]++;
        if (!ev.empty()) {
            std::vector<double>& back = dq.back();
            for (int s : ev)
                for (int64_t e = hpre[s]; e < hpre[s + 1]; e++)
                    back[(size_t)cpost[(size_t)e]] += cw[(size_t)e] * W_SYN;
        }
    }
    int win = std::max(1, T - BURN);
    if (state_leak > 0) {
        spk_v = v; spk_gg = gg; spk_adapt = adapt; spk_refr = refr;
        spk_dq0 = dq[0]; spk_dq1 = dq[1]; spk_nN = nN;
    }
    std::vector<float> hz(N);
    for (int i = 0; i < N; i++) hz[(size_t)i] = (float)(nsp[(size_t)i] + nrel[(size_t)i]) / win * 1000.0f;
    return hz;
}

} // namespace fly
