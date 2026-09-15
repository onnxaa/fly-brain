// core.cpp — FlyBrain load / encode / forward / step / train / sleep.
// Faithful port of fly_api.py pure-rate path (relu/lif), float32.
#include "fly.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace fly {

size_t file_size(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open: " + path);
    std::fseek(f, 0, SEEK_END);
    size_t n = (size_t)std::ftell(f);
    std::fclose(f);
    return n;
}
template <typename T>
static std::vector<T> load_raw(const std::string& path, size_t n) {
    std::vector<T> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open: " + path);
    size_t r = std::fread(v.data(), sizeof(T), n, f);
    std::fclose(f);
    if (r != n) throw std::runtime_error("short read: " + path);
    return v;
}
std::vector<int32_t> load_i32(const std::string& p, size_t n) { return load_raw<int32_t>(p, n); }
std::vector<float> load_f32(const std::string& p, size_t n) { return load_raw<float>(p, n); }
std::vector<uint8_t> load_u8(const std::string& p, size_t n) { return load_raw<uint8_t>(p, n); }
Stim stim_odor(const std::string& o) { Stim s; s.has_odor_str = true; s.odor_str = o; return s; }
Stim stim_alpn(const std::vector<float>& v) { Stim s; s.has_alpn = true; s.alpn = v; return s; }

FlyBrain::FlyBrain(const std::string& m, const std::string& path, int) { load(m, path); }

void FlyBrain::load(const std::string& m, const std::string& path) {
    mode = m; datapath = path;
    auto L32 = [&](const std::string& n) {
        return load_i32(path + "/" + n + ".i32", file_size(path + "/" + n + ".i32") / 4);
    };
    auto LF = [&](const std::string& n) {
        return load_f32(path + "/" + n + ".f32", file_size(path + "/" + n + ".f32") / 4);
    };
    std::string p = (mode == "mb") ? "mb" : "full";
    pre = L32(p + "_pre"); post = L32(p + "_post");
    std::vector<float> ws = LF(p + "_w");
    E = (int64_t)pre.size();
    wM.resize(E); sign.resize(E); wM0.resize(E);
    for (int64_t i = 0; i < E; i++) {
        float x = ws[i];
        sign[i] = (x >= 0) ? 1.0f : -1.0f;
        if (x == 0) sign[i] = 1.0f;
        wM[i] = std::fabs(x); wM0[i] = std::fabs(x);
    }
    auto need = [&](const std::string& n) { return L32(n); };
    if (mode == "mb") {
        ALPN = need("mb_inputs_ALPN"); KC = need("mb_KC"); MBON = need("mb_MBON");
        DAN = need("mb_DAN"); approach = need("mb_approach"); avoid = need("mb_avoid");
        dan_pam = need("mb_dan_pam"); dan_ppl = need("mb_dan_ppl");
        N = 0;
        for (auto v : pre) N = std::max(N, (int)v + 1);
        for (auto v : post) N = std::max(N, (int)v + 1);
        hops = 1;
        km_ki = need("mb_km_ki"); km_mi = need("mb_km_mi");
        km_is_avoid = load_u8(path + "/mb_km_is_avoid.u8",
                              file_size(path + "/mb_km_is_avoid.u8"));
        spike_ainc = 0.0f; spike_adapt = 1.0f; spike_burn = 0;
    } else {
        KC = need("full_KC"); MBON = need("full_MBON");
        approach = need("full_approach"); avoid = need("full_avoid");
        dan_pam = need("full_dan_pam"); dan_ppl = need("full_dan_ppl");
        ORN = need("full_ORN"); MECH = need("full_MECH"); VIS = need("full_VIS");
        ALPN = need("full_ALPN"); EFFERENT = need("full_EFFERENT");
        DESC = need("full_DESC"); MEv = need("full_ME"); LOv = need("full_LO");
        DESC_L = need("full_DESC_L"); DESC_R = need("full_DESC_R");
        ORN_L = need("full_ORN_L"); ORN_R = need("full_ORN_R");
        ALPN_L = need("full_ALPN_L"); ALPN_R = need("full_ALPN_R");
        MECH_L = need("full_MECH_L"); MECH_R = need("full_MECH_R");
        R = need("full_R"); VIS_eye = need("full_VIS_eye");
        R_cx = LF("full_R_cx"); R_cy = LF("full_R_cy");
        N = 138639;
        hops = 2;
        elig.assign(E, 0.0f);
        ref_in.assign(N, 0.0f);
        for (int64_t e = 0; e < E; e++) ref_in[post[(size_t)e]] += wM[(size_t)e];
        spike_ainc = 4.0f; spike_adapt = 0.2f; spike_burn = 50;
        build_km();
    }
    fan = LF(p + "_fan");
    if (mode == "mb") {
        std::vector<int32_t> sKC = KC, sMB = MBON;
        std::sort(sKC.begin(), sKC.end()); std::sort(sMB.begin(), sMB.end());
        km_e.clear(); km_e.reserve(km_ki.size());
        for (int64_t e = 0; e < E; e++)
            if (std::binary_search(sKC.begin(), sKC.end(), pre[(size_t)e]) &&
                std::binary_search(sMB.begin(), sMB.end(), post[(size_t)e]))
                km_e.push_back(e);
    }
    ref_mb.assign(MBON.size(), 0.0f);
    for (size_t i = 0; i < km_e.size(); i++) ref_mb[km_mi[i]] += wM[(size_t)km_e[i]];
    build_csr();
    // door odors (full)
    try {
        const char* doors[6] = {"geosmin","co2","hexanone3","methyl_salicylate","butanedione","ethyl_hexanoate"};
        for (auto d : doors) {
            std::string b = std::string("door_") + d;
            door_idx[d] = L32(b + "_idx");
            door_val[d] = LF(b + "_val");
        }
    } catch (...) {}
    try {
        taste_idx["sugar"] = L32("taste_sugar");
        taste_idx["bitter"] = L32("taste_bitter");
        taste_idx["water"] = L32("taste_water");
        taste_idx["ir94e"] = L32("taste_ir94e");
    } catch (...) {}
    try { odorA = L32("mb_odorA"); odorB = L32("mb_odorB"); } catch (...) {}
    // spike caches
    try {
        spike_apl = L32("spike_apl"); spike_gset = L32("spike_gset");
        if (mode == "mb") {
            sp_k2a_pre = L32("spike_k2a_pre"); sp_k2a_w = LF("spike_k2a_w");
            sp_k2a_apl = L32("spike_k2a_apl"); sp_a2k_post = L32("spike_a2k_post");
            sp_a2k_w = LF("spike_a2k_w"); sp_a2k_apl = L32("spike_a2k_apl");
        }
    } catch (...) {}
    try { clock_M = L32("clock_M"); clock_E = L32("clock_E"); } catch (...) {}
}

void FlyBrain::build_km() {
    std::vector<int32_t> sKC = KC, sMB = MBON;
    std::sort(sKC.begin(), sKC.end()); std::sort(sMB.begin(), sMB.end());
    std::vector<int32_t> KCsorted = KC, MBsorted = MBON;
    std::sort(KCsorted.begin(), KCsorted.end());
    std::sort(MBsorted.begin(), MBsorted.end());
    std::map<int32_t,int> kpos, mpos;
    for (size_t i = 0; i < KCsorted.size(); i++) kpos[KCsorted[i]] = (int)i;
    for (size_t i = 0; i < MBsorted.size(); i++) mpos[MBsorted[i]] = (int)i;
    km_ki.clear(); km_mi.clear(); km_is_avoid.clear(); km_e.clear();
    km_ki.reserve(256000); km_mi.reserve(256000); km_is_avoid.reserve(256000);
    std::vector<int32_t> savoid = avoid; std::sort(savoid.begin(), savoid.end());
    for (int64_t e = 0; e < E; e++) {
        int32_t pp = pre[(size_t)e], qq = post[(size_t)e];
        if (!std::binary_search(sKC.begin(), sKC.end(), pp)) continue;
        if (!std::binary_search(sMB.begin(), sMB.end(), qq)) continue;
        km_ki.push_back(kpos[pp]); km_mi.push_back(mpos[qq]);
        km_e.push_back(e);
        km_is_avoid.push_back(std::binary_search(savoid.begin(), savoid.end(), qq) ? 1 : 0);
    }
}

void FlyBrain::build_csr() {
    head.assign((size_t)N + 1, 0);
    for (int64_t e = 0; e < E; e++) head[(size_t)post[(size_t)e] + 1]++;
    for (int i = 0; i < N; i++) head[(size_t)i + 1] += head[(size_t)i];
    csr_pre.assign((size_t)E, 0); csr_sw.assign((size_t)E, 0);
    std::vector<int64_t> cur = head;
    for (int64_t e = 0; e < E; e++) {
        int64_t s = cur[(size_t)post[(size_t)e]]++;
        csr_pre[(size_t)s] = pre[(size_t)e];
        csr_sw[(size_t)s] = wM[(size_t)e] * sign[(size_t)e];
    }
    csr_built = true;
}

void FlyBrain::refresh_weights() { build_csr(); }

void FlyBrain::set_hops(int h) {
    if (h < 1 || h > 6) throw std::runtime_error("hops must be 1..6");
    hops = h;
}
int FlyBrain::get_hops() const { return hops; }

void FlyBrain::set_activation(const std::string& name, float sat, int Tms, int seed,
                              float wdrv, float ainc, float rmax, float adapt, int burn) {
    if (name != "relu" && name != "lif" && name != "spike")
        throw std::runtime_error("activation must be 'relu', 'lif' or 'spike'");
    act = name;
    if (name == "lif" && sat > 0) lif_sat = sat;
    if (name == "spike") {
        if (Tms >= 10 && Tms <= 5000) spike_T = Tms;
        spike_seed = seed; 
        if (wdrv > 0 && wdrv <= 200) spike_wdrv = wdrv;
        if (ainc >= 0) spike_ainc = ainc;
        if (rmax > 0 && rmax <= 500) spike_rmax = rmax;
        if (adapt >= 0) spike_adapt = adapt;
        if (burn >= 0) spike_burn = burn;
    }
}
std::map<std::string, float> FlyBrain::get_activation() const {
    return {{"lif_sat", lif_sat}, {"Tms", (float)spike_T}, {"seed", (float)spike_seed},
            {"wdrv", spike_wdrv}, {"ainc", spike_ainc}, {"rmax", spike_rmax},
            {"adapt", spike_adapt}, {"burn", (float)spike_burn}};
}

void FlyBrain::set_std(float alpha, float tau) {
    if (alpha < 0 || alpha > 1) throw std::runtime_error("std alpha must be 0..1");
    if (tau < 1) throw std::runtime_error("std tau must be >=1");
    std_alpha = alpha; std_tau = tau;
    if (alpha > 0 && (int)std_g.size() != N) {
        std::vector<float> ng((size_t)N, 1.0f);
        for (size_t i = 0; i < std_g.size() && i < ng.size(); i++) ng[i] = std_g[i];
        std_g.swap(ng);
    }
}
std::map<std::string, float> FlyBrain::get_std() const {
    return {{"alpha", std_alpha}, {"tau", std_tau}};
}
float FlyBrain::EI_sum() const {
    double s = 0;
    for (size_t i = 0; i < wM.size(); i++) s += (double)wM[i] * sign[i];
    return (float)s;
}
float FlyBrain::activate(float x) const {
    if (act == "lif") {
        if (x <= 0) return 0.0f;
        float s = lif_sat;
        return s * (1.0f - std::exp(-x / s));
    }
    return x > 0 ? x : 0.0f;
}

void FlyBrain::enable_auto_sleep(float k_wake, float thr_hi, float thr_lo,
                                 float night_lt, float crit_mult, float gate, float dose) {
    has_auto_sleep = true;
    sl_k_wake = k_wake; sl_thr_hi = thr_hi; sl_thr_lo = thr_lo;
    sl_night_lt = night_lt; sl_crit_mult = crit_mult; sl_gate = gate; sl_dose = dose;
}

std::pair<bool, std::map<std::string, float>> FlyBrain::sleep_tick(float kc_frac) {
    std::map<std::string, float> info;
    if (!has_auto_sleep) return {asleep, info};
    bool night;
    if (has_clock) night = ck_night;
    else night = ambient < sl_night_lt;
    if (asleep) {
        sleep_S = std::max(0.0f, sleep_S - sl_k_wake * 0.1f);
        if (sleep_S < sl_thr_lo) asleep = false;
    } else {
        sleep_S = sleep_S + sl_k_wake * kc_frac;
        if (sleep_S > sl_thr_hi && (night || sleep_S > sl_crit_mult * sl_thr_hi))
            asleep = true;
    }
    if (asleep && sl_dose > 0) {
        for (size_t i = 0; i < wM.size(); i++)
            wM[i] = wM[i] - sl_dose * (wM[i] - wM0[i]);
        build_csr();
    }
    info["sleep_S"] = sleep_S; info["asleep"] = asleep ? 1.0f : 0.0f;
    info["night"] = night ? 1.0f : 0.0f;
    return {asleep, info};
}

void FlyBrain::enable_scaling() {
    try {
        std::string n = (mode == "mb") ? "mb_fan" : "full_fan";
        std::string fp = datapath + "/" + n + ".f32";
        fan = load_f32(fp, file_size(fp) / 4);
    } catch (...) {}
}

void FlyBrain::calibrate(const std::vector<float>& image, int H, int Wd) {
    if (mode == "full") {
        if (!image.empty() && H > 0 && Wd > 0) {
            Stim s; s.has_image = true; s.image = image; s.imgH = H; s.imgW = Wd;
            step(s);
        } else {
            std::vector<float> g(64*64, 0.5f);
            Stim s; s.has_image = true; s.image = g; s.imgH = 64; s.imgW = 64;
            step(s);
        }
    } else {
        Stim s;
        step(s);
    }
}

// ---- encode ----
void FlyBrain::encode(const Stim& s, std::vector<int32_t>& idx, std::vector<float>& val,
                      std::map<std::string, float>& motion, bool& has_motion) {
    idx.clear(); val.clear(); motion.clear(); has_motion = false;
    if (s.has_image) {
        float m = 0;
        for (auto v : s.image) m += v;
        if (!s.image.empty()) m /= (float)s.image.size();
        ambient = m;
    }
    auto push_vec = [&](const std::vector<int32_t>& pool, const std::vector<float>& v, float scale) {
        size_t n = std::min(pool.size(), v.size());
        for (size_t i = 0; i < n; i++) { idx.push_back(pool[i]); val.push_back(v[i]*scale); }
    };
    // odor string
    if (s.has_odor_str) {
        const std::string& o = s.odor_str;
        if (o=="sugar"||o=="bitter"||o=="water"||o=="ir94e") {
            if (mode != "full") throw std::runtime_error("GRN tastes require mode='full'");
            auto it = taste_idx.find(o);
            if (it == taste_idx.end()) throw std::runtime_error("missing taste table");
            for (auto id : it->second) { idx.push_back(id); val.push_back(2.0f); }
        } else if (door_idx.count(o)) {
            if (mode != "full") throw std::runtime_error("DoOR requires mode='full'");
            auto& di = door_idx[o]; auto& dv = door_val[o];
            for (size_t i = 0; i < di.size(); i++) { idx.push_back(di[i]); val.push_back(dv[i]*2.0f); }
        } else if (o=="A"||o=="B") {
            if (mode=="full" && !odorA.empty() && !odorB.empty()) {
                auto& sel = (o=="A"?odorA:odorB);
                for (auto id : sel) { idx.push_back(id); val.push_back(2.0f); }
            } else {
                const std::vector<int32_t>& pool = (mode=="full"?ORN:ALPN);
                size_t half = pool.size()/2;
                if (o=="A") { for (size_t i=0;i<half;i++){idx.push_back(pool[i]);val.push_back(2.0f);} }
                else { for (size_t i=half;i<pool.size();i++){idx.push_back(pool[i]);val.push_back(2.0f);} }
            }
        } else {
            // unknown string -> B-half (matches Python fallback sel=o[half:])
            const std::vector<int32_t>& pool = (mode=="full"?ORN:ALPN);
            size_t half = pool.size()/2;
            for (size_t i=half;i<pool.size();i++){idx.push_back(pool[i]);val.push_back(2.0f);}
        }
    }
    if (s.has_odor_vec && !s.odor_vec.empty()) {
        const std::vector<int32_t>& tgt = (!ORN.empty()?ORN:ALPN);
        size_t n = std::min(tgt.size(), s.odor_vec.size());
        for (size_t i = 0; i < n; i++) { idx.push_back(tgt[i]); val.push_back(s.odor_vec[i]*2.0f); }
    }
    if (s.has_alpn && !s.alpn.empty()) {
        size_t n = std::min(ALPN.size(), s.alpn.size());
        for (size_t i = 0; i < n; i++) { idx.push_back(ALPN[i]); val.push_back(s.alpn[i]*2.0f); }
    }
    // lateral odors
    auto lateral_door = [&](const std::string& od, const std::vector<int32_t>& side) {
        auto it = door_idx.find(od);
        if (it == door_idx.end()) throw std::runtime_error("lateral odor: DoOR only");
        auto& di = it->second; auto& dv = door_val[od];
        std::unordered_set<int32_t> ss(side.begin(), side.end());
        for (size_t i = 0; i < di.size(); i++)
            if (ss.count(di[i])) { idx.push_back(di[i]); val.push_back(dv[i]*2.0f); }
    };
    if (!s.odor_left_str.empty() || !s.odor_right_str.empty() || s.has_odor_left || s.has_odor_right) {
        if (mode != "full" || ORN_L.empty())
            throw std::runtime_error("lateral odors require mode='full' with ORN_L/R");
        if (!s.odor_left_str.empty()) lateral_door(s.odor_left_str, ORN_L);
        if (!s.odor_right_str.empty()) lateral_door(s.odor_right_str, ORN_R);
        if (s.has_odor_left && !s.odor_left.empty()) {
            size_t n = std::min(ORN_L.size(), s.odor_left.size());
            for (size_t i=0;i<n;i++){idx.push_back(ORN_L[i]);val.push_back(s.odor_left[i]*2.0f);}
        }
        if (s.has_odor_right && !s.odor_right.empty()) {
            size_t n = std::min(ORN_R.size(), s.odor_right.size());
            for (size_t i=0;i<n;i++){idx.push_back(ORN_R[i]);val.push_back(s.odor_right[i]*2.0f);}
        }
    }
    if (s.has_mech && !s.mech.empty() && !MECH.empty()) {
        size_t n = std::min(MECH.size(), s.mech.size());
        for (size_t i = 0; i < n; i++) { idx.push_back(MECH[i]); val.push_back(s.mech[i]*2.0f); }
    }
    if ((s.has_mech_left && !s.mech_left.empty()) || (s.has_mech_right && !s.mech_right.empty())) {
        if (mode != "full" || MECH_L.empty())
            throw std::runtime_error("lateral mech requires mode='full' with MECH_L/R");
        if (s.has_mech_left) {
            size_t n = std::min(MECH_L.size(), s.mech_left.size());
            for (size_t i=0;i<n;i++){idx.push_back(MECH_L[i]);val.push_back(s.mech_left[i]*2.0f);}
        }
        if (s.has_mech_right) {
            size_t n = std::min(MECH_R.size(), s.mech_right.size());
            for (size_t i=0;i<n;i++){idx.push_back(MECH_R[i]);val.push_back(s.mech_right[i]*2.0f);}
        }
    }
    if (s.has_image) {
        if (mode != "full" || R.empty())
            throw std::runtime_error("images require mode='full' with R");
        std::vector<float> g = gray(s.image, s.imgH, s.imgW);
        std::vector<float> rv = sample_R(g, s.imgH, s.imgW);
        for (size_t i = 0; i < R.size(); i++) { idx.push_back(R[i]); val.push_back(rv[i]*2.0f); }
        auto sm = small16(g, s.imgH, s.imgW);
        auto pr = motion_energies(sm);
        motion = pr.first; has_motion = true;
    }
    if (has_clock) {
        float m = ck_morning;
        for (auto id : clock_M) { idx.push_back(id); val.push_back(clock_amp*(2*m-1)); }
        for (auto id : clock_E) { idx.push_back(id); val.push_back(clock_amp*(1-2*m)); }
    }
    if (asleep && has_auto_sleep) {
        for (auto& v : val) v = v * sl_gate;
    }
}

// ---- forward pure ----
std::vector<float> FlyBrain::forward_pure(const std::vector<int32_t>& idx,
                                          const std::vector<float>& val,
                                          int hh, float thr) {
    if ((int)fan.size() < N) fan.resize((size_t)N, 1.0f);
    std::vector<float> base((size_t)N, 0.0f), a((size_t)N, 0.0f);
    for (size_t i = 0; i < idx.size(); i++) {
        int32_t id = idx[i];
        if (id >= 0 && id < N) base[(size_t)id] += val[i];
    }
    a = base;
    std::vector<float> agg((size_t)N);
    for (int h = 0; h < hh; h++) {
        std::fill(agg.begin(), agg.end(), 0.0f);
        #pragma omp parallel for schedule(static) if(N>10000)
        for (int q = 0; q < N; q++) {
            float s = 0;
            for (int64_t e = head[(size_t)q]; e < head[(size_t)q+1]; e++)
                s += a[(size_t)csr_pre[(size_t)e]] * csr_sw[(size_t)e];
            agg[(size_t)q] = s;
        }
        for (int q = 0; q < N; q++) {
            float f = (q < (int)fan.size() && fan[(size_t)q] != 0) ? fan[(size_t)q] : 1.0f;
            float v = agg[(size_t)q] / (f + 1e-6f);
            a[(size_t)q] = activate(v - thr) + base[(size_t)q];
        }
    }
    return a;
}

static std::map<int32_t,int> sorted_pos(const std::vector<int32_t>& ids) {
    std::vector<int32_t> s = ids;
    std::sort(s.begin(), s.end());
    std::map<int32_t,int> m;
    for (size_t i = 0; i < s.size(); i++) m[s[i]] = (int)i;
    return m;
}

Out FlyBrain::step(const Stim& s, int hops_o, float thr) {
    int hh = (hops_o < 0) ? hops : hops_o;
    std::vector<int32_t> idx; std::vector<float> vv;
    std::map<std::string,float> mo; bool hm = false;
    encode(s, idx, vv, mo, hm);
    // STD (expression only)
    std::vector<float> vuse = vv;
    if (std_alpha > 0) {
        if ((int)std_g.size() != N) {
            std::vector<float> ng((size_t)N, 1.0f);
            for (size_t i=0;i<std_g.size()&&i<ng.size();i++) ng[i]=std_g[i];
            std_g.swap(ng);
        }
        for (size_t i = 0; i < std_g.size(); i++) std_g[i] += (1.0f - std_g[i]) / std_tau;
        for (size_t i = 0; i < idx.size(); i++) {
            int32_t id = idx[i];
            if (id < 0 || id >= N) continue;
            float orig = vv[i];
            float aval = orig/2.0f; if (aval<0) aval=0; if (aval>1) aval=1;
            vuse[i] = orig * std_g[(size_t)id];
            float ng = std_g[(size_t)id] * (1.0f - std_alpha * aval);
            if (ng<0) ng=0; if (ng>1) ng=1;
            std_g[(size_t)id] = ng;
        }
    }
    bool spk = (act == "spike");
    std::vector<float> h = spk ? forward_spike(idx, vuse) : forward_pure(idx, vuse, hh, thr);
    // KC top-5%
    size_t nKC = KC.size();
    std::vector<float> kcs(nKC);
    for (size_t i = 0; i < nKC; i++) kcs[i] = h[(size_t)KC[i]];
    std::vector<float> sk = kcs;
    size_t kk = std::max<size_t>(1, nKC*5/100);
    std::nth_element(sk.begin(), sk.begin() + (sk.size()-kk), sk.end());
    float kt = sk[sk.size()-kk];
    std::vector<char> m(nKC, 0);
    std::vector<float> ks(nKC, 0.0f);
    int kc_active = 0;
    if (spk) {
        for (size_t i = 0; i < nKC; i++) if (h[(size_t)KC[i]] > 0) kc_active++;
        for (size_t i = 0; i < nKC; i++) ks[i] = kcs[i];
    } else {
        for (size_t i = 0; i < nKC; i++) if (kcs[i] >= kt) { m[i]=1; kc_active++; ks[i]=kcs[i]; }
    }
    std::vector<float> r(MBON.size(), 0.0f);
    if (spk) {
        for (size_t i = 0; i < MBON.size(); i++) {
            // MBON order in file is sorted already for mb; for full KC/MBON sorted.
            // r index = position in MBON vector order. km_mi uses sorted order,
            // which matches file order when file is sorted. Map via sorted pos:
            r[i] = 0; // filled below via km
        }
        // Build via km_mi (sorted order). Need MBON sorted mapping: assume MBON file sorted.
        // To be safe, accumulate then reorder: accumulate into sorted-order array then map.
        std::vector<int32_t> sMB = MBON; std::sort(sMB.begin(), sMB.end());
        std::map<int32_t,int> mp; for (size_t i=0;i<sMB.size();i++) mp[sMB[i]]=(int)i;
        std::vector<float> rs(sMB.size(), 0.0f);
        // spike path: r = h[MBON] directly (measured Hz)
        for (size_t i = 0; i < MBON.size(); i++) {
            int pos = mp[MBON[i]];
            rs[(size_t)pos] = h[(size_t)MBON[i]];
        }
        // reorder back to MBON vector order
        for (size_t i = 0; i < MBON.size(); i++) r[i] = rs[(size_t)mp[MBON[i]]];
    } else {
        // analytic K->M readout; km_mi indexes sorted MBON order.
        std::vector<int32_t> sMB = MBON; std::sort(sMB.begin(), sMB.end());
        std::map<int32_t,int> mp; for (size_t i=0;i<sMB.size();i++) mp[sMB[i]]=(int)i;
        std::vector<double> rs(sMB.size(), 0.0);
        for (size_t j = 0; j < km_e.size(); j++) {
            int ki = km_ki[j], mi = km_mi[j];
            float kval = (ki>=0&&(size_t)ki<m.size()&&m[(size_t)ki]) ? ks[(size_t)ki] : 0.0f;
            if (kval==0) {
                // spike path already handled; for rate, ks already masked.
                // But ks is zero for non-top, so skip.
                // Note: ks[ki] corresponds to KC position ki (sorted KC order).
                // m[ki] mask aligns.
            }
            rs[(size_t)mi] += (double)kval * wM[(size_t)km_e[j]];
        }
        // map sorted-order rs back to MBON vector order
        for (size_t i = 0; i < MBON.size(); i++) r[i] = (float)rs[(size_t)mp[MBON[i]]];
    }
    auto mbpos = sorted_pos(MBON);
    auto mean_sel = [&](const std::vector<int32_t>& sel)->float {
        if (sel.empty()) return 0;
        double ssum=0; int c=0;
        // need r in sorted-order? Build sorted-order array for means:
        std::vector<int32_t> sMB = MBON; std::sort(sMB.begin(), sMB.end());
        std::map<int32_t,int> mp; for (size_t i=0;i<sMB.size();i++) mp[sMB[i]]=(int)i;
        // reconstruct sorted r:
        std::vector<float> rsorted(sMB.size(),0);
        for (size_t i=0;i<MBON.size();i++) rsorted[(size_t)mp[MBON[i]]]=r[i];
        for (auto id: sel){ auto it=mbpos.find(id); if(it==mbpos.end()) continue; ssum+=rsorted[(size_t)it->second]; c++; }
        return c? (float)(ssum/c):0;
    };
    float app = mean_sel(approach), avo = mean_sel(avoid);
    Out o;
    o.MB_app = app; o.MB_avo = avo; o.MB_pref = app - avo;
    o.MBON = r; o.KC_active = kc_active;
    auto mean_pool = [&](const std::vector<int32_t>& pool)->float {
        if (pool.empty()) return 0;
        double ssum=0; for (auto id: pool) if(id>=0&&id<N) ssum+=h[(size_t)id];
        return (float)(ssum/pool.size());
    };
    o.DAN_pam = mean_pool(dan_pam); o.DAN_ppl = mean_pool(dan_ppl);
    o.ALPN_mean = mean_pool(ALPN);
    o.EI_sum = EI_sum();
    if (mode=="full") {
        o.VIS_mean = mean_pool(VIS);
        if (!ALPN_L.empty()&&!ALPN_R.empty()&&!ALPN.empty()){
            o.ALPN_L=mean_pool(ALPN_L); o.ALPN_R=mean_pool(ALPN_R);
            o.turn_olf=o.ALPN_L-o.ALPN_R; o.has_lateral=true;
        }
        if (!ORN_L.empty()&&!ORN_R.empty()){ o.ORN_L=mean_pool(ORN_L); o.ORN_R=mean_pool(ORN_R); }
        o.MECH_mean=mean_pool(MECH);
        if(!MECH_L.empty()&&!MECH_R.empty()){o.MECH_L=mean_pool(MECH_L);o.MECH_R=mean_pool(MECH_R);}
        o.ORN_mean=mean_pool(ORN);
        if(!EFFERENT.empty()){
            std::vector<int32_t> eff=EFFERENT; std::sort(eff.begin(),eff.end());
            size_t he=eff.size()/2; double a=0,b=0;
            for(size_t i=0;i<he;i++) a+=h[(size_t)eff[i]];
            for(size_t i=he;i<eff.size();i++) b+=h[(size_t)eff[i]];
            if(he) a/=he; if(eff.size()-he) b/=(eff.size()-he);
            o.motor_pref=(float)(a-b); o.EFFERENT.assign(eff.size(),0);
            for(size_t i=0;i<eff.size();i++) o.EFFERENT[i]=h[(size_t)eff[i]];
        }
        if(!DESC_L.empty()&&!DESC_R.empty()){
            o.DN_L=mean_pool(DESC_L); o.DN_R=mean_pool(DESC_R);
            o.turn=o.DN_L-o.DN_R; o.has_dn=true;
        }
        if(!MEv.empty()){o.ME_mean=mean_pool(MEv);o.has_me=true;}
        if(hm){o.motion=mo;o.has_motion=true;}
    }
    if (has_auto_sleep) {
        float frac = nKC? (float)kc_active/(float)nKC : 0;
        auto pr = sleep_tick(frac);
        o.asleep = pr.first; o.sleep_S = sleep_S;
        auto it = pr.second.find("night");
        o.night = (it!=pr.second.end()&&it->second>0.5f);
    }
    if (!x_outputs.empty()) {
        std::vector<float> actv = h;
        // sparse KC code
        for (size_t i = 0; i < nKC; i++) {
            int32_t gid = KC[i];
            // KC[i] order vs sorted positions: m/ks indexed by file order, but
            // top-k threshold kt computed on file order — consistent.
            // Map file-order mask to global: actv[gid] = ks[i] if m else 0 (rate) or keep (spike).
            if (spk) actv[(size_t)gid] = h[(size_t)gid];
            else actv[(size_t)gid] = m[i]? ks[i]:0.0f;
        }
        for (auto& kv : x_outputs) {
            const XOut& xo = kv.second;
            std::vector<double> resp((size_t)xo.nn, 0.0);
            for (size_t i = 0; i < xo.epre.size(); i++) {
                int32_t pr2 = xo.epre[i]; int sl = xo.eslot[i];
                double w = (double)wM[(size_t)xo.ei[i]];
                resp[(size_t)sl] += (double)actv[(size_t)pr2] * w;
            }
            double mn=0; for(auto v:resp) mn+=v; if(!resp.empty()) mn/=resp.size();
            o.X[kv.first]=(float)mn;
            std::vector<float> all(resp.size());
            for(size_t i=0;i<resp.size();i++) all[i]=(float)resp[i];
            o.Xall[kv.first]=all;
        }
    }
    return o;
}

Out FlyBrain::train(const Stim& s, float reward, float punish, bool gated, int hops_o, float thr) {
    int hh = (hops_o < 0) ? hops : hops_o;
    std::vector<int32_t> idx; std::vector<float> vv;
    std::map<std::string,float> mo; bool hm=false;
    encode(s, idx, vv, mo, hm);
    bool spk = (act=="spike");
    std::vector<float> h = spk ? forward_spike(idx, vv) : forward_pure(idx, vv, hh, thr);
    size_t nKC = KC.size();
    std::vector<float> kcs(nKC);
    for (size_t i=0;i<nKC;i++) kcs[i]=h[(size_t)KC[i]];
    std::vector<float> sk=kcs;
    size_t kk=std::max<size_t>(1,nKC*5/100);
    std::nth_element(sk.begin(), sk.begin()+(sk.size()-kk), sk.end());
    float kt=sk[sk.size()-kk];
    std::vector<char> m(nKC,0);
    for(size_t i=0;i<nKC;i++) if(kcs[i]>=kt) m[i]=1;
    if (mode=="full" && reward!=0) {
        for (size_t e=0;e<(size_t)E;e++){
            float a1=h[(size_t)pre[e]], a2=h[(size_t)post[e]];
            elig[e]=elig[e]*0.9f + a1*a2;
        }
        for (size_t e=0;e<(size_t)E;e++){
            float dw = 0.002f*reward*elig[e];
            float lo=-0.02f*wM[e], hi=0.02f*wM[e];
            if(dw<lo)dw=lo; if(dw>hi)dw=hi;
            float nw=wM[e]+dw-1e-6f; if(nw<0.05f)nw=0.05f; if(nw>650.0f)nw=650.0f;
            wM[e]=nw;
        }
        if ((int)ref_in.size()<N) ref_in.assign((size_t)N,1.0f);
        std::vector<float> cur((size_t)N,0.0f);
        for (size_t e=0;e<(size_t)E;e++) cur[(size_t)post[e]]+=wM[e];
        std::vector<float> sc((size_t)N,1.0f);
        for(int i=0;i<N;i++) if(cur[(size_t)i]>1e-9f){
            float v=ref_in[(size_t)i]/cur[(size_t)i];
            if(v<0.95f)v=0.95f; if(v>1.05f)v=1.05f; sc[(size_t)i]=v;
        }
        for(size_t e=0;e<(size_t)E;e++) wM[e]=wM[e]*sc[(size_t)post[e]];
    }
    // gated DAN weights per KM edge
    std::vector<float> fkc;
    bool use_fkc=false;
    if (gated && !x_codes.empty() && s.has_odor_str) {
        std::vector<std::vector<int>> others;
        for(auto& kv: x_codes) if(kv.first!=s.odor_str) others.push_back(kv.second);
        fkc.assign(km_ki.size(), 0.85f);
        if(!others.empty()){
            std::vector<std::unordered_set<int>> sets;
            for(auto& o: others) sets.emplace_back(o.begin(), o.end());
            for(size_t j=0;j<km_ki.size();j++){
                int cnt=0;
                for(auto& st: sets) if(st.count(km_ki[(size_t)j])) cnt++;
                if(cnt>1)cnt=1;
                fkc[j]=0.85f+0.15f*(float)cnt;
            }
        }
        use_fkc=true;
    }
    if (reward>0) {
        for(size_t j=0;j<km_e.size();j++){
            if(!km_is_avoid[j]) continue;
            int ki=km_ki[j];
            if(ki<0||(size_t)ki>=m.size()||!m[(size_t)ki]) continue;
            int64_t e=km_e[j];
            wM[(size_t)e]=wM[(size_t)e]*(use_fkc?fkc[j]:0.85f);
        }
    }
    if (punish>0) {
        for(size_t j=0;j<km_e.size();j++){
            if(km_is_avoid[j]) continue;
            int ki=km_ki[j];
            if(ki<0||(size_t)ki>=m.size()||!m[(size_t)ki]) continue;
            int64_t e=km_e[j];
            wM[(size_t)e]=wM[(size_t)e]*(use_fkc?fkc[j]:0.85f);
        }
    }
    for(auto e: km_e) if(wM[(size_t)e]<0.05f) wM[(size_t)e]=0.05f;
    {
        std::vector<float> cur(MBON.size(),0.0f);
        for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
        std::vector<float> sc(MBON.size(),1.0f);
        for(size_t i=0;i<cur.size();i++) if(cur[i]>1e-9f){
            float v=ref_mb[i]/cur[i]; if(v<0.9f)v=0.9f; if(v>1.1f)v=1.1f; sc[i]=v;
        }
        for(size_t j=0;j<km_e.size();j++) wM[(size_t)km_e[j]]=(wM[(size_t)km_e[j]]*sc[(size_t)km_mi[j]]);
    }
    build_csr();
    return step(s, hh, thr);
}

void FlyBrain::sleep(int episodes, float rate) {
    for(int k=0;k<episodes;k++)
        for(size_t i=0;i<wM.size();i++) wM[i]=wM[i]-rate*(wM[i]-wM0[i]);
    std::vector<float> cur(MBON.size(),0.0f);
    for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
    ref_mb=cur;
    if(mode=="full"){
        if((int)ref_in.size()<N) ref_in.assign((size_t)N,0.0f);
        std::fill(ref_in.begin(), ref_in.end(), 0.0f);
        for(size_t e=0;e<(size_t)E;e++) ref_in[(size_t)post[e]]+=wM[e];
    }
    build_csr();
}

void FlyBrain::save_wbin(const std::string& path) {
    FILE* f=std::fopen(path.c_str(),"wb");
    if(!f) throw std::runtime_error("save open: "+path);
    std::fwrite(wM.data(), sizeof(float), wM.size(), f);
    std::fclose(f);
}
void FlyBrain::load_wbin(const std::string& path) {
    size_t n=file_size(path)/4;
    if(n!=wM.size()) throw std::runtime_error("wbin size mismatch");
    FILE* f=std::fopen(path.c_str(),"rb");
    if(!f) throw std::runtime_error("load open: "+path);
    size_t r=std::fread(wM.data(), sizeof(float), n, f);
    std::fclose(f);
    if(r!=n) throw std::runtime_error("short read wbin");
    std::vector<float> cur(MBON.size(),0.0f);
    for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
    ref_mb=cur;
    build_csr();
}

} // namespace fly
