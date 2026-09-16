// core.cpp — FlyBrain load / encode / forward / step / train / sleep.
// Faithful port of fly_api.py pure-rate path (relu/lif), float32.
#include "fly.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    std::string p = (mode == "mb") ? "mb" : (mode == "full") ? "full" :
                    (mode == "banc") ? "banc" : "male";
    if (mode != "mb" && mode != "full" && mode != "banc" && mode != "mcns")
        throw std::runtime_error("mode must be mb|full|banc|mcns");
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
    } else if (mode == "full") {
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
    } else {
        // whole-CNS (banc/mcns): files prefixed banc_* / male_*
        KC = need(p+"_KC"); MBON = need(p+"_MBON");
        approach = need(p+"_approach"); avoid = need(p+"_avoid");
        dan_pam = need(p+"_dan_pam"); dan_ppl = need(p+"_dan_ppl");
        ORN = need(p+"_ORN"); MECH = need(p+"_MECH");
        DESC_L = need(p+"_DESC_L"); DESC_R = need(p+"_DESC_R");
        ORN_L = need(p+"_ORN_L"); ORN_R = need(p+"_ORN_R");
        MECH_L = need(p+"_MECH_L"); MECH_R = need(p+"_MECH_R");
        MOT = need(p+"_MOTOR"); MOT_legL = need(p+"_leg_L");
        MOT_legR = need(p+"_leg_R"); MOT_wing = need(p+"_wing");
        MOT_neck = need(p+"_neck");
        try {
            CX_EPGv = need(p+"_CX_EPG"); CX_PFL = need(p+"_CX_PFL");
            CX_PFL_L = need(p+"_CX_PFL_L"); CX_PFL_R = need(p+"_CX_PFL_R");
        } catch (...) {}
        try { CX_wedge = need(p+"_CX_EPG_wedge"); } catch (...) {}
        R = need(p+"_R");
        try { R_cx = LF(p+"_R_cx"); R_cy = LF(p+"_R_cy"); } catch (...) {}
        try { R_L = L32(p+"_R_L"); R_R = L32(p+"_R_R"); } catch (...) {}
        try {
            Rret = L32(p+"_Rret"); Rret_cx = LF(p+"_Rret_cx"); Rret_cy = LF(p+"_Rret_cy");
            std::unordered_set<int32_t> rm(Rret.begin(), Rret.end());
            for (auto id : R_L) if (!rm.count(id)) R_LU.push_back(id);
            for (auto id : R_R) if (!rm.count(id)) R_RU.push_back(id);
        } catch (...) {}
        try {
            L1v = L32(p+"_L1"); L1cx = LF(p+"_L1_cx"); L1cy = LF(p+"_L1_cy");
            L2v = L32(p+"_L2"); L2cx = LF(p+"_L2_cx"); L2cy = LF(p+"_L2_cy");
        } catch (...) {}
        N = (int)LF(p + "_fan").size();
        hops = 2;
        elig.assign(E, 0.0f);
        ref_in.assign(N, 0.0f);
        for (int64_t e = 0; e < E; e++) ref_in[post[(size_t)e]] += wM[(size_t)e];
        spike_ainc = 4.0f; spike_adapt = 0.2f; spike_burn = 50;
        // SFA calibration mirrors Python: dense male MB needs stronger adaptation
        if (mode == "mcns") spike_ainc = 16.0f;
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
    ensure_readout_cache();
    // door odors (full: door_*; banc/mcns: prefixed)
    try {
        const char* doors[6] = {"geosmin","co2","hexanone3","methyl_salicylate","butanedione","ethyl_hexanoate"};
        std::string dp = (mode == "full") ? "door" : p + "_door";
        for (auto d : doors) {
            std::string b = dp + "_" + d;
            door_idx[d] = L32(b + "_idx");
            door_val[d] = LF(b + "_val");
        }
    } catch (...) {}
    try {
        std::string tp = (mode == "full") ? "taste" : p + "_taste";
        taste_idx["sugar"] = L32(tp + "_sugar");
        taste_idx["bitter"] = L32(tp + "_bitter");
        taste_idx["water"] = L32(tp + "_water");
        taste_idx["ir94e"] = L32(tp + "_ir94e");
    } catch (...) {}
    try { odorA = L32("mb_odorA"); odorB = L32("mb_odorB"); } catch (...) {}
    // spike caches (per-mode APL + graded sets; FAFB files are wrong neurons elsewhere)
    try {
        std::string sp = (mode == "mb" || mode == "full") ? "spike" : p + "_spike";
        spike_apl = L32(sp + "_apl"); spike_gset = L32(sp + "_gset");
        if (mode == "mb") {
            sp_k2a_pre = L32("spike_k2a_pre"); sp_k2a_w = LF("spike_k2a_w");
            sp_k2a_apl = L32("spike_k2a_apl"); sp_a2k_post = L32("spike_a2k_post");
            sp_a2k_w = LF("spike_a2k_w"); sp_a2k_apl = L32("spike_a2k_apl");
        }
    } catch (...) {}
    try {
        std::string cp = (mode == "full") ? "clock" : p + "_clock";
        clock_M = L32(cp + "_M"); clock_E = L32(cp + "_E");
    } catch (...) {}
    // real VNC (MANC): data present => has_vnc_data; runs only after enable_vnc()
    try {
        vpre = L32("vnc_pre"); vpost = L32("vnc_post");
        std::vector<float> vws = LF("vnc_w");
        vE = (int64_t)vpre.size();
        vwM.resize((size_t)vE); vsign.resize((size_t)vE);
        for (int64_t i = 0; i < vE; i++) {
            float x = vws[i];
            vsign[(size_t)i] = (x >= 0) ? 1.0f : -1.0f;
            if (x == 0) vsign[(size_t)i] = 1.0f;
            vwM[(size_t)i] = std::fabs(x);
        }
        vfan = LF("vnc_fan");
        vN = (int)vfan.size();
        vdesc = L32("vnc_desc"); vmotor = L32("vnc_motor_all");
        vlegL = L32("vnc_leg_L"); vlegR = L32("vnc_leg_R");
        vwingL = L32("vnc_wing_L"); vwingR = L32("vnc_wing_R");
        vneck = L32("vnc_neck");
        std::vector<int32_t> bb = L32("vnc_bridge_b"), bv = L32("vnc_bridge_v");
        vbridge.assign((size_t)vN, {});
        for (size_t i = 0; i < bb.size() && i < bv.size(); i++) {
            int32_t v = bv[i];
            if (v >= 0 && v < vN) vbridge[(size_t)v].push_back(bb[i]);
        }
        // CSR-direct: edges stably sorted by post (export_vnc.py) => counting head
        vhead.assign((size_t)vN + 1, 0);
        for (int64_t i = 0; i < vE; i++) {
            int32_t q = vpost[(size_t)i];
            if (q >= 0 && q < vN) vhead[(size_t)q + 1]++;
        }
        for (int i = 0; i < vN; i++) vhead[(size_t)i + 1] += vhead[(size_t)i];
        vcsr_pre.assign((size_t)vE, 0); vcsr_sw.assign((size_t)vE, 0.0f);
        std::vector<int64_t> cur(vhead.begin(), vhead.begin() + vN);
        for (int64_t i = 0; i < vE; i++) {
            int32_t q = vpost[(size_t)i];
            if (q < 0 || q >= vN) continue;
            int64_t s = cur[(size_t)q]++;
            vcsr_pre[(size_t)s] = vpre[(size_t)i];
            vcsr_sw[(size_t)s] = vwM[(size_t)i] * vsign[(size_t)i];
        }
        has_vnc_data = true;
    } catch (...) { has_vnc_data = false; }
}

void FlyBrain::enable_vnc(float bridge_w) {
    if (!has_vnc_data) throw std::runtime_error("no VNC data (run export_vnc.py)");
    if (mode != "full") throw std::runtime_error("VNC needs mode='full'");
    vnc_on = true;
    vbridge_w = bridge_w;
    x_log.push_back({"enable_vnc", "", "manc121"});
}

std::vector<float> FlyBrain::forward_vnc(const std::vector<float>& h_brain) {
    int n = vN;
    if ((int)v_base.size() < n) {
        v_base.assign((size_t)n, 0.0f);
        v_a.assign((size_t)n, 0.0f);
        v_agg.assign((size_t)n, 0.0f);
    }
    std::fill(v_base.begin(), v_base.begin() + n, 0.0f);
    if (vbridge_w != 0.0f) {
        for (int32_t v : vdesc) {
            if (v < 0 || v >= n) continue;
            const auto& bl = vbridge[(size_t)v];
            if (bl.empty()) continue;
            double s = 0;
            for (int32_t b : bl)
                if (b >= 0 && (size_t)b < h_brain.size()) s += h_brain[(size_t)b];
            float m = (float)(s / bl.size()) * vbridge_w;
            if (m > 0) v_base[(size_t)v] = m;
        }
    }
    v_a = v_base;
    for (int h = 0; h < 2; h++) {
        #pragma omp parallel for schedule(static) if(n > 10000)
        for (int q = 0; q < n; q++) {
            float s = 0;
            for (int64_t e = vhead[(size_t)q]; e < vhead[(size_t)q + 1]; e++)
                s += v_a[(size_t)vcsr_pre[(size_t)e]] * vcsr_sw[(size_t)e];
            v_agg[(size_t)q] = s;
        }
        #pragma omp parallel for schedule(static) if(n > 10000)
        for (int q = 0; q < n; q++) {
            float x = v_agg[(size_t)q] / (vfan[(size_t)q] + 1e-6f);
            v_a[(size_t)q] = (x > 0 ? x : 0.0f) + v_base[(size_t)q];
        }
    }
    return v_a;
}

void FlyBrain::build_km() {
    // O(E) bitmap version (was: 2x binary_search per edge).
    // Same sets as Python: KC/MBON membership, KC rank in sorted-KC order,
    // MB rank in sorted-MBON order, avoid flag by MBON id.
    std::vector<int32_t> KCsorted = KC, MBsorted = MBON;
    std::sort(KCsorted.begin(), KCsorted.end());
    std::sort(MBsorted.begin(), MBsorted.end());
    std::vector<int> kpos(N, -1), mpos(N, -1);
    for (size_t i = 0; i < KCsorted.size(); i++) kpos[(size_t)KCsorted[i]] = (int)i;
    for (size_t i = 0; i < MBsorted.size(); i++) mpos[(size_t)MBsorted[i]] = (int)i;
    std::vector<char> isK((size_t)N, 0), isM((size_t)N, 0), isAv((size_t)N, 0);
    for (auto id : KC) if (id >= 0 && id < N) isK[(size_t)id] = 1;
    for (auto id : MBON) if (id >= 0 && id < N) isM[(size_t)id] = 1;
    for (auto id : avoid) if (id >= 0 && id < N) isAv[(size_t)id] = 1;
    km_ki.clear(); km_mi.clear(); km_is_avoid.clear(); km_e.clear();
    km_ki.reserve(256000); km_mi.reserve(256000); km_is_avoid.reserve(256000);
    for (int64_t e = 0; e < E; e++) {
        int32_t pp = pre[(size_t)e], qq = post[(size_t)e];
        if (pp < 0 || pp >= N || qq < 0 || qq >= N) continue;
        if (!isK[(size_t)pp] || !isM[(size_t)qq]) continue;
        km_ki.push_back(kpos[(size_t)pp]); km_mi.push_back(mpos[(size_t)qq]);
        km_e.push_back(e);
        km_is_avoid.push_back(isAv[(size_t)qq] ? 1 : 0);
    }
}

void FlyBrain::ensure_readout_cache() {
    mb_sorted = MBON;
    std::sort(mb_sorted.begin(), mb_sorted.end());
    std::map<int32_t,int> mp;
    for (size_t i = 0; i < mb_sorted.size(); i++) mp[mb_sorted[i]] = (int)i;
    mb_index.assign(MBON.size(), 0);
    for (size_t i = 0; i < MBON.size(); i++) {
        auto it = mp.find(MBON[i]);
        mb_index[i] = (it == mp.end()) ? 0 : it->second;
    }
    ai_pos.clear(); vi_pos.clear();
    for (auto id : approach) { auto it = mp.find(id); if (it != mp.end()) ai_pos.push_back(it->second); }
    for (auto id : avoid) { auto it = mp.find(id); if (it != mp.end()) vi_pos.push_back(it->second); }
    eff_sorted = EFFERENT;
    std::sort(eff_sorted.begin(), eff_sorted.end());
}

void FlyBrain::build_csr() {
    // Fast path: edges stably sorted by post (export_flat.py) => edge index
    // IS the CSR slot. O(E) scan to verify + sequential fills, no scatter,
    // no permutation map. Within-post order == raw parquet order (stable),
    // so per-post FP summation sequences are unchanged (parity-safe).
    bool sorted = true;
    for (int64_t e = 1; e < E; e++) {
        if (post[(size_t)e] < post[(size_t)e - 1]) { sorted = false; break; }
    }
    head.assign((size_t)N + 1, 0);
    for (int64_t e = 0; e < E; e++) head[(size_t)post[(size_t)e] + 1]++;
    for (int i = 0; i < N; i++) head[(size_t)i + 1] += head[(size_t)i];
    csr_sw.assign((size_t)E, 0);
    if (sorted) {
        csr_direct = true;
        edge2csr.clear();
        csr_pre = pre; // identical order: copy (sequential 60MB)
        const float* w = wM.data(); const float* sg = sign.data();
        float* sw = csr_sw.data();
        int64_t n = E;
        #pragma omp parallel for schedule(static) if(n>1000000)
        for (int64_t e = 0; e < n; e++) sw[(size_t)e] = w[(size_t)e] * sg[(size_t)e];
    } else {
        csr_direct = false;
        csr_pre.assign((size_t)E, 0);
        edge2csr.assign((size_t)E, 0);
        std::vector<int64_t> cur = head;
        for (int64_t e = 0; e < E; e++) {
            int64_t s = cur[(size_t)post[(size_t)e]]++;
            csr_pre[(size_t)s] = pre[(size_t)e];
            csr_sw[(size_t)s] = wM[(size_t)e] * sign[(size_t)e];
            edge2csr[(size_t)e] = (int32_t)s;
        }
    }
    csr_built = true;
}

void FlyBrain::refresh_weights() {
    // Allocation-free weight refresh (topology unchanged): same values as
    // build_csr would produce. Direct layout: single sequential pass.
    if (csr_direct) {
        if (csr_sw.size() != (size_t)E) { build_csr(); return; }
        const float* w = wM.data(); const float* sg = sign.data();
        float* sw = csr_sw.data();
        int64_t n = E;
        #pragma omp parallel for schedule(static) if(n>1000000)
        for (int64_t e = 0; e < n; e++) sw[(size_t)e] = w[(size_t)e] * sg[(size_t)e];
        return;
    }
    if (edge2csr.size() != (size_t)E || csr_sw.size() != (size_t)E) { build_csr(); return; }
    const float* w = wM.data(); const float* sg = sign.data();
    float* sw = csr_sw.data(); const int32_t* m = edge2csr.data();
    int64_t n = E;
    #pragma omp parallel for schedule(static) if(n>1000000)
    for (int64_t e = 0; e < n; e++) sw[(size_t)m[(size_t)e]] = w[(size_t)e] * sg[(size_t)e];
}

void FlyBrain::csr_update_edge(int64_t e) {
    size_t s = csr_direct ? (size_t)e : (size_t)edge2csr[(size_t)e];
    csr_sw[s] = wM[(size_t)e] * sign[(size_t)e];
}

void FlyBrain::set_hops(int h) {
    if (h < 1 || h > 6) throw std::runtime_error("hops must be 1..6");
    hops = h;
}
int FlyBrain::get_hops() const { return hops; }

void FlyBrain::set_scaling(const std::string& s) {
    if (s != "static" && s != "active") throw std::runtime_error("scaling must be 'static' or 'active'");
    scaling = (s == "active") ? 1 : 0;
}
std::string FlyBrain::get_scaling() const { return scaling ? "active" : "static"; }

float FlyBrain::set_state(float leak) {
    if (leak < 0 || leak >= 1) throw std::runtime_error("leak must be in [0,1)");
    state_leak = leak;
    if ((int)f_prev.size() != N) f_prev.assign((size_t)N, 0.0f);
    return state_leak;
}
void FlyBrain::reset_state() { f_prev.assign((size_t)N, 0.0f); }

void FlyBrain::set_activation(const std::string& name, float sat, int Tms, int seed,
                              float wdrv, float ainc, float rmax, float adapt, int burn) {
    if (name != "relu" && name != "lif" && name != "spike")
        throw std::runtime_error("activation must be 'relu', 'lif' or 'spike'");
    act = name;
    act_id = (name == "lif") ? 1 : 0;
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
        int64_t n = (int64_t)wM.size();
        #pragma omp parallel for schedule(static) if(n>1000000)
        for (int64_t i = 0; i < n; i++) {
            size_t ee = (size_t)i;
            float nw = wM[ee] - sl_dose * (wM[ee] - wM0[ee]);
            wM[ee] = nw;
            csr_update_edge(i);
        }
    }
    info["sleep_S"] = sleep_S; info["asleep"] = asleep ? 1.0f : 0.0f;
    info["night"] = night ? 1.0f : 0.0f;
    return {asleep, info};
}

void FlyBrain::enable_scaling() {
    try {
        std::string n = (mode == "mb") ? "mb_fan" : (mode == "full") ? "full_fan" :
                        (mode == "banc") ? "banc_fan" : "male_fan";
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
            if (mode != "full" && mode != "banc" && mode != "mcns") throw std::runtime_error("GRN tastes need a CNS mode");
            auto it = taste_idx.find(o);
            if (it == taste_idx.end()) throw std::runtime_error("missing taste table");
            for (auto id : it->second) { idx.push_back(id); val.push_back(2.0f); }
        } else if (door_idx.count(o)) {
            if (mode != "full" && mode != "banc" && mode != "mcns") throw std::runtime_error("DoOR requires a CNS mode");
            auto& di = door_idx[o]; auto& dv = door_val[o];
            for (size_t i = 0; i < di.size(); i++) { idx.push_back(di[i]); val.push_back(dv[i]*2.0f); }
        } else if (o=="A"||o=="B") {
            if (mode=="full" && !odorA.empty() && !odorB.empty()) {
                auto& sel = (o=="A"?odorA:odorB);
                for (auto id : sel) { idx.push_back(id); val.push_back(2.0f); }
            } else {
                const std::vector<int32_t>& pool = (!ORN.empty()?ORN:ALPN);
                size_t half = pool.size()/2;
                if (o=="A") { for (size_t i=0;i<half;i++){idx.push_back(pool[i]);val.push_back(2.0f);} }
                else { for (size_t i=half;i<pool.size();i++){idx.push_back(pool[i]);val.push_back(2.0f);} }
            }
        } else {
            // unknown string -> B-half (matches Python fallback sel=o[half:])
            const std::vector<int32_t>& pool = (!ORN.empty()?ORN:ALPN);
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
        if (ORN_L.empty())
            throw std::runtime_error("lateral odors require ORN_L/R pools");
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
        if (MECH_L.empty())
            throw std::runtime_error("lateral mech requires MECH_L/R pools");
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
        if (R.empty())
            throw std::runtime_error("images require R photoreceptors (not mb)");
        if (s.vpol < 0 || s.vpol > 2) throw std::runtime_error("vpol must be 0/1/2 (lum/on/off)");
        std::vector<float> g = gray(s.image, s.imgH, s.imgW);
        bool lonly = (mode == "banc" && s.vpol != 0);
        if (!R_cx.empty() && R_cx.size() == R.size() && !lonly) {
            std::vector<float> rv = sample_R(g, s.imgH, s.imgW);
            for (size_t i = 0; i < R.size(); i++) { idx.push_back(R[i]); val.push_back(rv[i]*2.0f); }
        } else if (!Rret.empty() && Rret_cx.size() == Rret.size()) {
            // homology-anchored RF (mcns) + eye-mean for unmapped
            std::vector<float> rr = sample_at(g, s.imgH, s.imgW, Rret_cx, Rret_cy);
            for (size_t i = 0; i < Rret.size(); i++) { idx.push_back(Rret[i]); val.push_back(rr[i]*2.0f); }
            double lv = 0, rv2 = 0;
            for (int y = 0; y < s.imgH; y++)
                for (int x = 0; x < s.imgW; x++) {
                    float v = g[(size_t)y * s.imgW + x];
                    if (x < s.imgW / 2) lv += v; else rv2 += v;
                }
            lv /= (s.imgH * (s.imgW / 2)); rv2 /= (s.imgH * (s.imgW - s.imgW / 2));
            for (auto id : R_LU) { idx.push_back(id); val.push_back((float)lv * 2.0f); }
            for (auto id : R_RU) { idx.push_back(id); val.push_back((float)rv2 * 2.0f); }
        } else if (!R_L.empty() && !R_R.empty() && R_cx.empty() && Rret.empty()) {
            // honest eye split, only when no RF map exists at all (else the
            // extra R drive would inhibit the L prosthesis via HisCl synapses)
            double lv = 0, rv2 = 0;
            for (int y = 0; y < s.imgH; y++)
                for (int x = 0; x < s.imgW; x++) {
                    float v = g[(size_t)y * s.imgW + x];
                    if (x < s.imgW / 2) lv += v; else rv2 += v;
                }
            lv /= (s.imgH * (s.imgW / 2)); rv2 /= (s.imgH * (s.imgW - s.imgW / 2));
            for (auto id : R_L) { idx.push_back(id); val.push_back((float)lv * 2.0f); }
            for (auto id : R_R) { idx.push_back(id); val.push_back((float)rv2 * 2.0f); }
        } else if (!lonly) throw std::runtime_error("images: no R_cx map and no R_L/R split");
        // vpol: 0=lum raw to both (legacy), 1=on (L1 increments),
        // 2=off (L2 decrements), bg=0.5
        std::vector<float> g1 = g, g2 = g;
        if (s.vpol == 1) {
            for (auto& v : g1) v = v > 0.5f ? v - 0.5f : 0.0f;
            std::fill(g2.begin(), g2.end(), 0.0f);
        } else if (s.vpol == 2) {
            std::fill(g1.begin(), g1.end(), 0.0f);
            for (auto& v : g2) v = v < 0.5f ? 0.5f - v : 0.0f;
        }
        if (!L1v.empty() && L1cx.size() == L1v.size()) {
            // luminance proxy (R1-6 absent in BANC): L1/L2 at column RF
            std::vector<float> l1 = sample_at(g1, s.imgH, s.imgW, L1cx, L1cy);
            for (size_t i = 0; i < L1v.size(); i++) { idx.push_back(L1v[i]); val.push_back(l1[i] * 2.0f); }
        } else if (s.vpol == 1 && !L1v.empty()) {
            double m = 0; for (auto v : g1) m += v; m /= g1.size();
            for (auto id : L1v) { idx.push_back(id); val.push_back((float)m * 2.0f); }
        }
        if (!L2v.empty() && L2cx.size() == L2v.size()) {
            std::vector<float> l2 = sample_at(g2, s.imgH, s.imgW, L2cx, L2cy);
            for (size_t i = 0; i < L2v.size(); i++) { idx.push_back(L2v[i]); val.push_back(l2[i] * 2.0f); }
        } else if (s.vpol == 2 && !L2v.empty()) {
            double m = 0; for (auto v : g2) m += v; m /= g2.size();
            for (auto id : L2v) { idx.push_back(id); val.push_back((float)m * 2.0f); }
        }
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
    if ((int)f_base.size() < N) {
        f_base.assign((size_t)N, 0.0f);
        f_a.assign((size_t)N, 0.0f);
        f_agg.assign((size_t)N, 0.0f);
        f_fan.assign((size_t)N, 1.0f);
    }
    std::fill(f_base.begin(), f_base.begin() + N, 0.0f);
    for (size_t i = 0; i < idx.size(); i++) {
        int32_t id = idx[i];
        if (id >= 0 && id < N) f_base[(size_t)id] += val[i];
    }
    if ((int)f_prev.size() != N) f_prev.assign((size_t)N, 0.0f);
    if (state_leak > 0 && (int)f_prev.size() == N)
        for (int i = 0; i < N; i++) f_base[(size_t)i] += state_leak * f_prev[(size_t)i];
    f_a = f_base; // copy (size N, buffers exact)
    const float* __restrict__ sw = csr_sw.data();
    const int32_t* __restrict__ cp = csr_pre.data();
    const int64_t* __restrict__ hd = head.data();
    const float* __restrict__ fn = fan.data();
    const float* __restrict__ bs = f_base.data();
    float* __restrict__ av = f_a.data();
    float* __restrict__ ag = f_agg.data();
    float* __restrict__ af = f_fan.data();
    int n = N;
    bool big = (n > 10000);
    bool active = (scaling == 1);
    if (act_id == 1) {
        float sat = lif_sat;
        for (int h = 0; h < hh; h++) {
            #pragma omp parallel for schedule(static) if(big)
            for (int q = 0; q < n; q++) {
                float s = 0, f = 0;
                for (int64_t e = hd[q]; e < hd[q + 1]; e++) {
                    float a = av[(size_t)cp[(size_t)e]];
                    s += a * sw[(size_t)e];
                    if (active && a > 0) f += std::fabs(sw[(size_t)e]);
                }
                ag[q] = s; af[q] = f;
            }
            #pragma omp parallel for schedule(static) if(big)
            for (int q = 0; q < n; q++) {
                float denom = active ? af[q] : (fn[q] + 1e-6f);
                float v = (active && af[q] <= 1e-9f) ? 0.0f : ag[q] / denom - thr;
                av[q] = (v > 0 ? sat * (1.0f - std::exp(-v / sat)) : 0.0f) + bs[q];
            }
        }
    } else {
        for (int h = 0; h < hh; h++) {
            #pragma omp parallel for schedule(static) if(big)
            for (int q = 0; q < n; q++) {
                float s = 0, f = 0;
                for (int64_t e = hd[q]; e < hd[q + 1]; e++) {
                    float a = av[(size_t)cp[(size_t)e]];
                    s += a * sw[(size_t)e];
                    if (active && a > 0) f += std::fabs(sw[(size_t)e]);
                }
                ag[q] = s; af[q] = f;
            }
            #pragma omp parallel for schedule(static) if(big)
            for (int q = 0; q < n; q++) {
                float denom = active ? af[q] : (fn[q] + 1e-6f);
                float v = (active && af[q] <= 1e-9f) ? 0.0f : ag[q] / denom - thr;
                av[q] = (v > 0 ? v : 0.0f) + bs[q];
            }
        }
    }
    f_prev = f_a;
    return f_a;
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
    // rs accumulates in mb_sorted order (km_mi convention); r follows
    // MBON vector order via mb_index (precomputed in load).
    std::vector<float> rsorted(mb_sorted.size(), 0.0f);
    if (spk) {
        // spike path: measured Hz at MBON directly
        for (size_t i = 0; i < MBON.size(); i++)
            rsorted[(size_t)mb_index[i]] = h[(size_t)MBON[i]];
    } else {
        // analytic K->M readout
        std::vector<double> racc(mb_sorted.size(), 0.0);
        for (size_t j = 0; j < km_e.size(); j++) {
            int ki = km_ki[j], mi = km_mi[j];
            float kval = (ki>=0&&(size_t)ki<m.size()&&m[(size_t)ki]) ? ks[(size_t)ki] : 0.0f;
            racc[(size_t)mi] += (double)kval * wM[(size_t)km_e[j]];
        }
        for (size_t i = 0; i < rsorted.size(); i++) rsorted[i] = (float)racc[i];
    }
    for (size_t i = 0; i < MBON.size(); i++) r[i] = rsorted[(size_t)mb_index[i]];
    auto mean_sorted = [&](const std::vector<int>& poss)->float {
        if (poss.empty()) return 0;
        double ssum = 0;
        for (int p : poss) ssum += rsorted[(size_t)p];
        return (float)(ssum / poss.size());
    };
    float app = mean_sorted(ai_pos), avo = mean_sorted(vi_pos);
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
    if (mode=="full" || mode=="banc" || mode=="mcns") {
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
            const std::vector<int32_t>& eff = eff_sorted;
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
        if(!CX_EPGv.empty()){
            double se=0; for (auto id: CX_EPGv) if(id>=0&&id<N) se+=h[(size_t)id];
            o.CX_EPG=(float)(se/CX_EPGv.size());
            if(CX_wedge.size()==CX_EPGv.size() && !CX_EPGv.empty()){
                int bi=0; float bv=-1;
                for(size_t i=0;i<CX_wedge.size();i++){
                    int32_t id=CX_EPGv[(size_t)CX_wedge[i]];
                    float v=(id>=0&&id<N)?h[(size_t)id]:0;
                    if(v>bv){bv=v;bi=(int)i;}
                }
                o.CX_bump=bi;
            } else o.CX_bump=-1;
            if(!CX_PFL_L.empty()&&!CX_PFL_R.empty()){
                o.CX_PFL_L=mean_pool(CX_PFL_L); o.CX_PFL_R=mean_pool(CX_PFL_R);
                o.CX_turn=o.CX_PFL_L-o.CX_PFL_R;
            }
            o.has_cx=true;
        }
        if(!MOT.empty()){
            o.BANC_motor=mean_pool(MOT);
            o.BANC_leg_L=mean_pool(MOT_legL); o.BANC_leg_R=mean_pool(MOT_legR);
            o.BANC_leg_imb=o.BANC_leg_L-o.BANC_leg_R;
            o.BANC_wing=mean_pool(MOT_wing); o.BANC_neck=mean_pool(MOT_neck);
            o.has_bcmotor=true;
        }
        if (vnc_on && has_vnc_data) {
            std::vector<float> hv = forward_vnc(h);
            auto vmean = [&](const std::vector<int32_t>& pool)->float {
                if (pool.empty()) return 0;
                double s = 0; for (auto id : pool) if (id >= 0 && id < vN) s += hv[(size_t)id];
                return (float)(s / pool.size());
            };
            o.VNC_desc_mean = vmean(vdesc);
            o.VNC_motor = vmean(vmotor);
            o.VNC_leg_L = vmean(vlegL); o.VNC_leg_R = vmean(vlegR);
            o.VNC_leg_imb = o.VNC_leg_L - o.VNC_leg_R;
            o.VNC_wing_L = vmean(vwingL); o.VNC_wing_R = vmean(vwingR);
            o.VNC_wing = (o.VNC_wing_L + o.VNC_wing_R) * 0.5f;
            o.VNC_neck = vmean(vneck);
            o.has_vnc = true;
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

Out FlyBrain::train(const Stim& s, float reward, float punish, bool gated, int hops_o, float thr, bool stdp) {
    int hh = (hops_o < 0) ? hops : hops_o;
    std::vector<int32_t> idx; std::vector<float> vv;
    std::map<std::string,float> mo; bool hm=false;
    encode(s, idx, vv, mo, hm);
    bool spk = (act=="spike");
    std::vector<float> h = spk ? forward_spike(idx, vv, stdp) : forward_pure(idx, vv, hh, thr);
    if (spk && stdp) refresh_weights();
    size_t nKC = KC.size();
    std::vector<float> kcs(nKC);
    for (size_t i=0;i<nKC;i++) kcs[i]=h[(size_t)KC[i]];
    std::vector<float> sk=kcs;
    size_t kk=std::max<size_t>(1,nKC*5/100);
    std::nth_element(sk.begin(), sk.begin()+(sk.size()-kk), sk.end());
    float kt=sk[sk.size()-kk];
    std::vector<char> m(nKC,0);
    for(size_t i=0;i<nKC;i++) if(kcs[i]>=kt) m[i]=1;
    if ((mode=="full"||mode=="banc"||mode=="mcns") && reward!=0 && !(spk && stdp)) {
        // fused eligibility + weight update (was: two passes over E)
        int64_t nE = E;
        #pragma omp parallel for schedule(static) if(nE>1000000)
        for (int64_t e = 0; e < nE; e++) {
            size_t ee = (size_t)e;
            float a1=h[(size_t)pre[ee]], a2=h[(size_t)post[ee]];
            float el = elig[ee]*0.9f + a1*a2;
            elig[ee] = el;
            float dw = 0.002f*reward*el;
            float lo=-0.02f*wM[ee], hi=0.02f*wM[ee];
            if(dw<lo)dw=lo; if(dw>hi)dw=hi;
            float nw=wM[ee]+dw-1e-6f; if(nw<0.05f)nw=0.05f; if(nw>650.0f)nw=650.0f;
            wM[ee]=nw;
            csr_update_edge(ee);
        }
        if ((int)ref_in.size()<N) ref_in.assign((size_t)N,1.0f);
        std::vector<float> cur((size_t)N,0.0f);
        for (size_t e=0;e<(size_t)E;e++) cur[(size_t)post[e]]+=wM[e];
        std::vector<float> sc((size_t)N,1.0f);
        for(int i=0;i<N;i++) if(cur[(size_t)i]>1e-9f){
            float v=ref_in[(size_t)i]/cur[(size_t)i];
            if(v<0.95f)v=0.95f; if(v>1.05f)v=1.05f; sc[(size_t)i]=v;
        }
        int64_t nE2 = E;
        #pragma omp parallel for schedule(static) if(nE2>1000000)
        for (int64_t e = 0; e < nE2; e++) {
            size_t ee = (size_t)e;
            float nw = wM[ee]*sc[(size_t)post[ee]];
            wM[ee]=nw;
            csr_update_edge(ee);
        }
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
            float nw=wM[(size_t)e]*(use_fkc?fkc[j]:0.85f);
            wM[(size_t)e]=nw;
            csr_update_edge(e);
        }
    }
    if (punish>0) {
        for(size_t j=0;j<km_e.size();j++){
            if(km_is_avoid[j]) continue;
            int ki=km_ki[j];
            if(ki<0||(size_t)ki>=m.size()||!m[(size_t)ki]) continue;
            int64_t e=km_e[j];
            float nw=wM[(size_t)e]*(use_fkc?fkc[j]:0.85f);
            wM[(size_t)e]=nw;
            csr_update_edge(e);
        }
    }
    for(auto e: km_e) {
        if(wM[(size_t)e]<0.05f) { wM[(size_t)e]=0.05f; csr_update_edge(e); }
    }
    {
        std::vector<float> cur(MBON.size(),0.0f);
        for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
        std::vector<float> sc(MBON.size(),1.0f);
        for(size_t i=0;i<cur.size();i++) if(cur[i]>1e-9f){
            float v=ref_mb[i]/cur[i]; if(v<0.9f)v=0.9f; if(v>1.1f)v=1.1f; sc[i]=v;
        }
        for(size_t j=0;j<km_e.size();j++) {
            int64_t e=km_e[j];
            wM[(size_t)e]=(wM[(size_t)e]*sc[(size_t)km_mi[j]]);
            csr_update_edge(e);
        }
    }
    return step(s, hh, thr);
}

void FlyBrain::sleep(int episodes, float rate) {
    // fused wash + CSR write-through (was: wash passes + full CSR rebuild)
    for(int k=0;k<episodes;k++) {
        int64_t n = (int64_t)wM.size();
        #pragma omp parallel for schedule(static) if(n>1000000)
        for (int64_t i = 0; i < n; i++) {
            size_t ee = (size_t)i;
            float nw = wM[ee]-rate*(wM[ee]-wM0[ee]);
            wM[ee]=nw;
            csr_update_edge(ee);
        }
    }
    std::vector<float> cur(MBON.size(),0.0f);
    for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
    ref_mb=cur;
    if(mode=="full"||mode=="banc"||mode=="mcns"){
        if((int)ref_in.size()<N) ref_in.assign((size_t)N,0.0f);
        std::fill(ref_in.begin(), ref_in.end(), 0.0f);
        for(size_t e=0;e<(size_t)E;e++) ref_in[(size_t)post[e]]+=wM[e];
    }
}

void FlyBrain::save_wbin(const std::string& path) {
    FILE* f=std::fopen(path.c_str(),"wb");
    if(!f) throw std::runtime_error("save open: "+path);
    // header: magic(8) + mode(16, nul-padded) + N(i64) + E(i64); legacy files stay loadable
    char magic[8] = {'F','L','Y','W','B','I','N','1'};
    char mbuf[16] = {0};
    std::snprintf(mbuf, sizeof(mbuf), "%s", mode.c_str());
    int64_t hh[2] = {(int64_t)N, (int64_t)E};
    std::fwrite(magic, 1, 8, f);
    std::fwrite(mbuf, 1, 16, f);
    std::fwrite(hh, sizeof(int64_t), 2, f);
    std::fwrite(wM.data(), sizeof(float), wM.size(), f);
    std::fclose(f);
}
void FlyBrain::load_wbin(const std::string& path) {
    size_t nb=file_size(path);
    FILE* f=std::fopen(path.c_str(),"rb");
    if(!f) throw std::runtime_error("load open: "+path);
    size_t n;
    char magic[8];
    if (nb >= 8 && std::fread(magic, 1, 8, f) == 8 &&
        std::memcmp(magic, "FLYWBIN1", 8) == 0) {
        char mbuf[16]; int64_t hh[2];
        if (std::fread(mbuf, 1, 16, f) != 16 || std::fread(hh, sizeof(int64_t), 2, f) != 2) {
            std::fclose(f); throw std::runtime_error("wbin: truncated header");
        }
        std::string fmode(mbuf, strnlen(mbuf, 16));
        if (fmode != mode)
            { std::fclose(f); throw std::runtime_error("wbin mode mismatch: file=" + fmode + " brain=" + mode); }
        if (hh[0] != (int64_t)N || hh[1] != (int64_t)E)
            { std::fclose(f); throw std::runtime_error("wbin N/E mismatch (X-growth changes E)"); }
        n = (nb - 40) / 4;
    } else {
        std::rewind(f);
        n = nb / 4;
    }
    if(n!=wM.size()) { std::fclose(f); throw std::runtime_error("wbin size mismatch"); }
    size_t r=std::fread(wM.data(), sizeof(float), n, f);
    std::fclose(f);
    if(r!=n) throw std::runtime_error("short read wbin");
    std::vector<float> cur(MBON.size(),0.0f);
    for(size_t j=0;j<km_e.size();j++) cur[(size_t)km_mi[j]]+=wM[(size_t)km_e[j]];
    ref_mb=cur;
    refresh_weights();
}

} // namespace fly
