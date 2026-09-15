// xzone.cpp — experimental zone 1:1 with fly_api.py x_* methods.
// RNG note: Python uses np.random.default_rng(seed) (PCG64); C++ uses
// mt19937_64. Edge samples differ bit-wise; behavior class is identical
// (same distributions, same counts, same update rules).
#include "fly.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace fly {

int64_t FlyBrain::x_add_edge(int32_t a, int32_t b, float w, float dale) {
    int64_t i0 = E;
    pre.push_back(a); post.push_back(b);
    wM.push_back(w); sign.push_back(dale); wM0.push_back(w);
    E++;
    if (mode == "full") {
        elig.push_back(0.0f);
        if ((int)fan.size() < N) fan.resize(N, 1.0f);
        if (b >= 0 && b < N) fan[b] += std::fabs(w);
        if ((int)ref_in.size() < N) ref_in.resize(N, 0.0f);
        if (b >= 0 && b < N) ref_in[b] += w;
    } else {
        if ((int)fan.size() < N) fan.resize(N, 1.0f);
    }
    x_log.push_back({"add_edge", "", std::to_string(a) + "->" + std::to_string(b)});
    build_csr();
    return i0;
}

std::vector<int32_t> FlyBrain::x_add_output(const std::string& name, int n,
                                            const std::vector<int32_t>& src_pool,
                                            int per_in, float wscale, uint64_t seed) {
    if (name.empty() || name[0] == '_') throw std::runtime_error("bad output name");
    if (x_outputs.count(name)) throw std::runtime_error("output exists: " + name);
    std::mt19937_64 rng(seed);
    std::vector<int32_t> src = src_pool.empty() ? KC : src_pool;
    if (src.empty()) throw std::runtime_error("src_pool is empty");
    if (per_in < 0) per_in = (mode == "mb") ? 8 : 64;
    if (wscale < 0) wscale = (mode == "mb") ? 0.05f : 0.5f;
    // weight distribution of src-driven edges (data), fallback all wM
    std::vector<float> dS;
    dS.reserve(1024);
    std::vector<int32_t> ssrc = src;
    std::sort(ssrc.begin(), ssrc.end());
    for (int64_t e = 0; e < E; e++)
        if (std::binary_search(ssrc.begin(), ssrc.end(), pre[e])) dS.push_back(wM[e]);
    if (dS.empty()) dS = wM;
    std::uniform_int_distribution<size_t> pickD(0, dS.size() - 1);
    std::vector<int32_t> pa, pb;
    std::vector<float> pw;
    std::vector<int32_t> nw;
    for (int k = 0; k < n; k++) nw.push_back(N + k);
    int m = std::min<int>(per_in, (int)src.size());
    std::vector<int32_t> pool = src;
    for (int k = 0; k < n; k++) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < m; j++) {
            pa.push_back(pool[(size_t)j]);
            pb.push_back(nw[(size_t)k]);
            pw.push_back(dS[pickD(rng)] * wscale);
        }
    }
    N += n;
    int64_t n0 = E;
    for (size_t i = 0; i < pa.size(); i++) {
        pre.push_back(pa[i]); post.push_back(pb[i]);
        wM.push_back(pw[i]); sign.push_back(1.0f); wM0.push_back(pw[i]);
        E++;
        if (mode == "full") elig.push_back(0.0f);
    }
    fan.resize(N, 1.0f);
    // (mb branch in Python leaves _fan short; C++ hardens with 1.0 — same
    // value a fresh enable_scaling would give an un-driven node.)
    if (mode == "full") {
        if ((int)ref_in.size() < N) ref_in.resize(N, 0.0f);
        for (size_t i = 0; i < pa.size(); i++) {
            fan[pb[i]] += std::fabs(pw[i]);
            ref_in[pb[i]] += pw[i];
        }
    }
    XOut xo;
    xo.name = name; xo.ids = nw; xo.nn = n;
    xo.ei.resize(pa.size()); xo.epre.resize(pa.size()); xo.eslot.resize(pa.size());
    std::map<int32_t, int> slot;
    for (int k = 0; k < n; k++) slot[nw[(size_t)k]] = k;
    for (size_t i = 0; i < pa.size(); i++) {
        xo.ei[i] = n0 + (int64_t)i;
        xo.epre[i] = pa[i];
        xo.eslot[i] = slot[pb[i]];
    }
    x_outputs[name] = xo;
    x_log.push_back({"add_output", name, "n=" + std::to_string(n)});
    build_csr();
    return nw;
}

std::map<std::string, int> FlyBrain::x_list_outputs() const {
    std::map<std::string, int> o;
    for (auto& kv : x_outputs) o[kv.first] = (int)kv.second.ids.size();
    return o;
}

std::map<std::string, float> FlyBrain::x_teach_output(
    const std::string& name, int trials, float eta, bool high, bool has_target,
    float target, const std::string& scope, float eta_full, const Stim& stim,
    int hops, float thr, std::vector<float>* hist) {
    auto it = x_outputs.find(name);
    if (it == x_outputs.end()) throw std::runtime_error("unknown output: " + name);
    XOut& xo = it->second;
    int hh = (hops < 0) ? this->hops : hops;
    std::vector<float> hst;
    bool spk = (act == "spike");
    for (int t = 0; t < std::max(1, trials); t++) {
        std::vector<int32_t> idx; std::vector<float> val;
        std::map<std::string, float> mo; bool hm = false;
        encode(stim, idx, val, mo, hm);
        std::vector<float> h = spk ? forward_spike(idx, val) : forward_pure(idx, val, hh, thr);
        // sparse KC code drives the update
        std::vector<float> kc(KC.size());
        for (size_t i = 0; i < KC.size(); i++) kc[i] = h[KC[i]];
        std::vector<float> skc = kc;
        size_t kk = std::max<size_t>(1, kc.size() * 5 / 100);
        std::nth_element(skc.begin(), skc.begin() + (skc.size() - kk), skc.end());
        float kt = skc[skc.size() - kk];
        std::map<int32_t, int> kpos;
        for (size_t i = 0; i < KC.size(); i++) kpos[KC[i]] = (int)i;
        for (auto& kv : kpos) h[kv.first] = (h[kv.first] >= kt) ? h[kv.first] : 0.0f;
        size_t L = xo.epre.size();
        std::vector<float> pre_act(L);
        float mx = 0;
        for (size_t i = 0; i < L; i++) {
            pre_act[i] = h[xo.epre[i]];
            mx = std::max(mx, pre_act[i]);
        }
        std::vector<float> pn(L, 0.0f);
        if (mx > 1e-9f)
            for (size_t i = 0; i < L; i++) pn[i] = pre_act[i] / mx;
        float err, cur = 0;
        if (has_target) {
            std::vector<double> resp(xo.nn, 0.0);
            for (size_t i = 0; i < L; i++)
                resp[(size_t)xo.eslot[i]] += (double)h[xo.epre[i]] * wM[(size_t)xo.ei[i]];
            cur = 0;
            for (auto v : resp) cur += (float)v;
            cur /= xo.nn;
            err = target - cur;
        } else if (high) { err = 1.0f; }
        else { err = -1.0f; }
        if (has_target) {
            std::vector<double> cnt(xo.nn, 0.0);
            for (size_t i = 0; i < L; i++)
                if (pn[i] > 1e-6f) cnt[(size_t)xo.eslot[i]] += 1.0;
            for (size_t i = 0; i < L; i++) {
                double den = cnt[(size_t)xo.eslot[i]] * mx;
                if (den > 1e-12) {
                    double st = eta * err * pn[i] / den;
                    int64_t e = xo.ei[i];
                    wM[(size_t)e] = std::min(650.0f, std::max(0.05f, (float)(wM[(size_t)e] + st)));
                }
            }
        } else if (high) {
            for (size_t i = 0; i < L; i++) {
                int64_t e = xo.ei[i];
                wM[(size_t)e] = std::min(650.0f, std::max(0.05f, wM[(size_t)e] * (1 + eta * pn[i])));
            }
        } else {
            for (size_t i = 0; i < L; i++) {
                int64_t e = xo.ei[i];
                wM[(size_t)e] = std::min(650.0f, std::max(0.05f, wM[(size_t)e] * (1 - eta * pn[i])));
            }
        }
        if (scope == "whole") {
            double mxd = mx > 1e-12 ? mx : 1.0;
            double ef = eta_full * err;
            if (ef != 0.0) {
                for (int64_t e = 0; e < E; e++) {
                    double c = (double)h[pre[(size_t)e]] * h[post[(size_t)e]] / (mxd * mxd);
                    double dw = std::min(0.02, std::max(-0.02, ef * c));
                    wM[(size_t)e] = std::min(650.0f, std::max(0.05f, (float)(wM[(size_t)e] * (1 + dw))));
                }
                build_csr();
            }
        }
        Out o = step(stim, hh, thr);
        auto f = o.X.find(name);
        hst.push_back(f == o.X.end() ? 0.0f : f->second);
    }
    x_log.push_back({"teach_output", name, "trials=" + std::to_string(trials)});
    if (hist) *hist = hst;
    return {{"final", hst.empty() ? 0.0f : hst.back()}};
}

void FlyBrain::x_remember(const std::string& odor) {
    Stim s = stim_odor(odor);
    std::vector<int32_t> idx; std::vector<float> val;
    std::map<std::string, float> mo; bool hm = false;
    encode(s, idx, val, mo, hm);
    std::vector<float> h = forward_pure(idx, val, hops, 0.0f);
    std::vector<float> kc(KC.size());
    for (size_t i = 0; i < KC.size(); i++) kc[i] = h[KC[i]];
    std::vector<float> skc = kc;
    size_t kk = std::max<size_t>(1, kc.size() * 5 / 100);
    std::nth_element(skc.begin(), skc.begin() + (skc.size() - kk), skc.end());
    float kt = skc[skc.size() - kk];
    std::vector<int> code;
    std::map<int32_t, int> kpos;
    for (size_t i = 0; i < KC.size(); i++) kpos[KC[i]] = (int)i;
    for (size_t i = 0; i < km_ki.size(); i++)
        if (h[KC[(size_t)km_ki[i]]] >= kt) code.push_back(km_ki[i]);
    x_codes[odor] = code;
}

std::map<std::string, double> FlyBrain::x_report() const {
    double wsum = 0, w0sum = 0;
    for (size_t i = 0; i < wM.size(); i++) {
        wsum += wM[i];
        w0sum += wM0[i];
    }
    double drift = (w0sum > 0) ? (wsum - w0sum) / w0sum * 100.0 : 0.0;
    return {{"edges", (double)E}, {"N", (double)N}, {"EI_drift_%", drift},
            {"ops", (double)x_log.size()}};
}

} // namespace fly