#pragma once
// fly.hpp — C++ twin of fly_api.py (FlyBrainAPI).
// Data-only whole-brain Drosophila: frozen topology/Dale, protocol-only learning.
// Flat binaries from export_flat.py / export_spike.py (no Arrow/parquet here).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fly {

struct Stim {
    std::string odor_str; bool has_odor_str = false;
    std::vector<float> odor_vec; bool has_odor_vec = false;
    std::vector<float> alpn;     bool has_alpn = false;
    std::vector<float> mech;     bool has_mech = false;
    // full-mode lateral (vectors over the side pool)
    std::vector<float> odor_left;  bool has_odor_left = false;
    std::string odor_left_str;
    std::vector<float> odor_right; bool has_odor_right = false;
    std::string odor_right_str;
    std::vector<float> mech_left;  bool has_mech_left = false;
    std::vector<float> mech_right; bool has_mech_right = false;
    std::vector<float> image; int imgH = 0, imgW = 0; bool has_image = false;
    int vpol = 0; // 0=lum legacy, 1=on (L1 increments), 2=off (L2 decrements)
};

struct Out {
    float MB_pref = 0, MB_app = 0, MB_avo = 0;
    std::vector<float> MBON;
    int KC_active = 0;
    float DAN_pam = 0, DAN_ppl = 0, ALPN_mean = 0, EI_sum = 0;
    // full extras
    float VIS_mean = 0, ALPN_L = 0, ALPN_R = 0, turn_olf = 0;
    float ORN_L = 0, ORN_R = 0, MECH_mean = 0, MECH_L = 0, MECH_R = 0;
    float ORN_mean = 0, motor_pref = 0, DN_L = 0, DN_R = 0, turn = 0, ME_mean = 0;
    bool has_lateral = false, has_dn = false, has_me = false;
    // whole-CNS (banc/mcns) intact motor readouts, valid when has_bcmotor
    float BANC_motor = 0, BANC_leg_L = 0, BANC_leg_R = 0, BANC_leg_imb = 0;
    float BANC_wing = 0, BANC_neck = 0;
    bool has_bcmotor = false;
    float CX_EPG = 0, CX_PFL_L = 0, CX_PFL_R = 0, CX_turn = 0;
    int CX_bump = -1; bool has_cx = false;
    // real VNC (MANC) readouts, valid when has_vnc
    float VNC_desc_mean = 0, VNC_motor = 0;
    float VNC_leg_L = 0, VNC_leg_R = 0, VNC_leg_imb = 0;
    float VNC_wing_L = 0, VNC_wing_R = 0, VNC_wing = 0, VNC_neck = 0;
    bool has_vnc = false;
    std::vector<float> EFFERENT;
    std::map<std::string, float> motion; bool has_motion = false;
    std::map<std::string, float> X;
    std::map<std::string, std::vector<float>> Xall;
    bool asleep = false; float sleep_S = 0; bool night = false;
};

struct XOut {
    std::string name;
    std::vector<int32_t> ids;
    std::vector<int64_t> ei;     // edge indices of incoming edges
    std::vector<int32_t> epre;   // presynaptic neuron per incoming edge
    std::vector<int32_t> eslot;  // output slot per incoming edge
    int nn = 0;
};

struct XLog { std::string op, name, extra; };

class FlyBrain {
public:
    std::string mode = "mb", datapath = ".";
    int N = 0; int64_t E = 0;
    // edges (frozen topology; wM magnitudes trainable, signs frozen)
    std::vector<int32_t> pre, post;
    std::vector<float> wM, sign, wM0, fan;
    // CSR by post for forward: head size N+1
    std::vector<int64_t> head;
    std::vector<int32_t> csr_pre;
    std::vector<float> csr_sw;
    // edge index -> CSR slot (rebuilt by build_csr; enables O(km) and
    // allocation-free weight refreshes without recounting head).
    // EMPTY = identity (CSR-direct layout: edges stably sorted by post,
    // edge index IS the CSR slot; see export_flat.py sorted_by_post).
    std::vector<int32_t> edge2csr;
    bool csr_direct = false;
    bool csr_built = false;
    // persistent forward buffers (avoid 3xN allocs per forward_pure call)
    std::vector<float> f_base, f_a, f_agg, f_fan, f_prev;
    // roles
    std::vector<int32_t> ORN, MECH, VIS, ALPN, EFFERENT, DESC, MEv, LOv;
    std::vector<int32_t> ORN_L, ORN_R, ALPN_L, ALPN_R, MECH_L, MECH_R;
    std::vector<int32_t> DESC_L, DESC_R, R, VIS_eye;
    std::vector<int32_t> R_L, R_R, L1v, L2v, Rret, R_LU, R_RU;
    std::vector<int32_t> CX_EPGv, CX_PFL, CX_PFL_L, CX_PFL_R, CX_wedge;
    std::vector<float> Rret_cx, Rret_cy;
    std::vector<float> L1cx, L1cy, L2cx, L2cy;
    // whole-CNS (banc/mcns) motor pools
    std::vector<int32_t> MOT, MOT_legL, MOT_legR, MOT_wing, MOT_neck;
    std::vector<float> R_cx, R_cy;
    std::vector<int32_t> KC, MBON, DAN, approach, avoid, dan_pam, dan_ppl;
    std::vector<int32_t> clock_M, clock_E;
    // readout index caches (built once in load; MBON/groups are immutable):
    // mb_sorted = sorted MBON ids; ai_pos/vi_pos = sorted-order positions of
    // approach/avoid; eff_sorted = sorted EFFERENT ids (motor split)
    std::vector<int32_t> mb_sorted;
    std::vector<int> ai_pos, vi_pos;
    std::vector<int32_t> eff_sorted;
    std::vector<int> mb_index; // mb_index[i] = position of MBON[i] in mb_sorted
    void ensure_readout_cache();
    // KC->MBON analytic readout maps
    std::vector<int64_t> km_e; // global edge index per KM entry
    std::vector<int32_t> km_ki, km_mi;
    std::vector<uint8_t> km_is_avoid;
    std::vector<float> ref_mb, ref_in, elig;
    // protocol state
    int hops = 1;
    int scaling = 0; // 0=static fan (default, all validations), 1=active fan (deep chains live)
    float state_leak = 0.0f; // inter-step carry, 0=off (bit-identical classic)
    std::string act = "relu"; float lif_sat = 2.0f;
    int act_id = 0; // 0=relu, 1=lif (mirrors act; avoids per-element strcmp)
    int spike_T = 200, spike_seed = 7; float spike_wdrv = 68.75f;
    float spike_ainc = 0.0f, spike_rmax = 150.0f, spike_adapt = 1.0f;
    int spike_burn = 0;
    float std_alpha = 0.0f, std_tau = 25.0f;
    std::vector<float> std_g;
    // sleep / clock
    bool has_auto_sleep = false;
    float sl_k_wake = 0.05f, sl_thr_hi = 1.0f, sl_thr_lo = 0.3f;
    float sl_night_lt = 0.25f, sl_crit_mult = 2.0f, sl_gate = 0.2f, sl_dose = 0.002f;
    float sleep_S = 0.0f; bool asleep = false; float ambient = 0.5f;
    bool has_clock = false; float clock_amp = 0.5f;
    // TTFL state (clock.cpp, Goldbeter 1995 1:1)
    double ck_M = 0.2, ck_P0 = 0.3, ck_P1 = 0.3, ck_P2 = 1.5, ck_PN = 0.5;
    double ck_t = 0.0, ck_avg = -1.0;
    bool ck_night = false; float ck_morning = 0.5f, ck_night_frac = 0.5f;
    // X zone
    std::map<std::string, XOut> x_outputs;
    std::map<std::string, std::vector<int>> x_in; // reserved
    std::vector<XLog> x_log;
    std::map<std::string, std::vector<int>> x_codes;
    // door / taste / odorA-B caches (flat export)
    std::map<std::string, std::vector<int32_t>> door_idx;
    std::map<std::string, std::vector<float>> door_val;
    std::map<std::string, std::vector<int32_t>> taste_idx;
    std::vector<int32_t> odorA, odorB;
    // spike caches
    std::vector<int32_t> spike_apl, spike_gset;
    std::vector<int32_t> sp_k2a_pre, sp_k2a_apl, sp_a2k_post, sp_a2k_apl;
    std::vector<float> sp_k2a_w, sp_a2k_w;
    // vision motion memory
    std::vector<float> last_small; bool has_last_small = false;
    // real VNC (MANC v1.2.1, opt-in via enable_vnc; full mode only).
    // Frozen topology/Dale from MANC; brain->VNC bridge is X-zone protocol.
    bool has_vnc_data = false, vnc_on = false;
    int vN = 0; int64_t vE = 0;
    std::vector<int32_t> vpre, vpost;
    std::vector<float> vwM, vsign, vfan;
    std::vector<int64_t> vhead;
    std::vector<int32_t> vcsr_pre;
    std::vector<float> vcsr_sw;
    std::vector<int32_t> vdesc, vmotor, vlegL, vlegR, vwingL, vwingR, vneck;
    std::vector<std::vector<int32_t>> vbridge; // per-VNC-idx brain idxs
    float vbridge_w = 1.0f;
    std::vector<float> v_base, v_a, v_agg;

    FlyBrain() = default;
    FlyBrain(const std::string& m, const std::string& path, int seed = 1);
    void load(const std::string& m, const std::string& path);
    void build_csr();
    void build_km(); // (re)build KC->MBON maps from current edges

    // protocol setters (mirror Python names)
    void set_hops(int h); int get_hops() const;
    void set_scaling(const std::string& s); std::string get_scaling() const;
    float set_state(float leak); void reset_state();
    void set_activation(const std::string& name, float sat = 2.0f, int Tms = 200,
                        int seed = 7, float wdrv = 68.75f, float ainc = -1,
                        float rmax = 150.0f, float adapt = -1, int burn = -1);
    std::map<std::string, float> get_activation() const;
    void set_std(float alpha, float tau);
    std::map<std::string, float> get_std() const;
    float EI_sum() const;
    void refresh_weights(); // update csr_sw + ref after wM change (no topology change)
    void csr_update_edge(int64_t e); // write-through of one edge weight
    void enable_auto_sleep(float k_wake = 0.05f, float thr_hi = 1.0f,
                           float thr_lo = 0.3f, float night_lt = 0.25f,
                           float crit_mult = 2.0f, float gate = 0.2f,
                           float dose = 0.002f);
    void enable_clock(float amp = 0.5f);
    std::map<std::string, double> tick_clock(double hours = 1.0, float light = -1.0f);
    void enable_scaling(); // loads exported fan (formula-equivalent)
    void enable_vnc(float bridge_w = 1.0f); // load MANC VNC + type bridge
    std::vector<float> forward_vnc(const std::vector<float>& h_brain);
    void calibrate(const std::vector<float>& image = {}, int H = 0, int Wd = 0);

    // core
    void encode(const Stim& s, std::vector<int32_t>& idx, std::vector<float>& val,
                std::map<std::string, float>& motion, bool& has_motion);
    std::vector<float> forward_pure(const std::vector<int32_t>& idx,
                                    const std::vector<float>& val,
                                    int hops, float thr);
    std::vector<float> forward_spike(const std::vector<int32_t>& idx,
                                     const std::vector<float>& val,
                                     bool plastic = false);
    double stdp_Aplus = 0.005, stdp_Aminus = 0.0052, stdp_tau = 20.0;
    Out step(const Stim& s, int hops = -1, float thr = 0.0f);
    Out train(const Stim& s, float reward = 0.0f, float punish = 0.0f,
              bool gated = true, int hops = -1, float thr = 0.0f,
              bool stdp = false);
    void sleep(int episodes = 1, float rate = 0.02f);
    void save_wbin(const std::string& path);
    void load_wbin(const std::string& path);

    // X zone
    std::vector<int32_t> x_add_output(const std::string& name, int n = 8,
                                      const std::vector<int32_t>& src_pool = {},
                                      int per_in = -1, float wscale = -1,
                                      uint64_t seed = 0);
    std::map<std::string, int> x_list_outputs() const;
    std::map<std::string, float> x_teach_output(const std::string& name,
                                                int trials = 3, float eta = 0.3f,
                                                bool high = true, bool has_target = false,
                                                float target = 0.0f,
                                                const std::string& scope = "new",
                                                float eta_full = 0.02f,
                                                const Stim& stim = Stim(),
                                                int hops = -1, float thr = 0.0f,
                                                std::vector<float>* hist = nullptr);
    void x_remember(const std::string& odor);
    std::map<std::string, double> x_report() const;
    int64_t x_add_edge(int32_t a, int32_t b, float w, float dale = 1.0f);

    // internal
    float activate(float x) const;
    std::pair<bool, std::map<std::string, float>> sleep_tick(float kc_frac);
    std::vector<float> gray(const std::vector<float>& img, int H, int Wd) const;
    std::vector<float> sample_R(const std::vector<float>& g, int H, int Wd) const;
    std::vector<float> sample_at(const std::vector<float>& g, int H, int Wd,
                                 const std::vector<float>& cxs,
                                 const std::vector<float>& cys) const;
    std::vector<float> small16(const std::vector<float>& g, int H, int Wd) const;
    std::pair<std::map<std::string, float>, bool> motion_energies(
        const std::vector<float>& small);
};

// free helpers
std::vector<int32_t> load_i32(const std::string& path, size_t n);
std::vector<float> load_f32(const std::string& path, size_t n);
std::vector<uint8_t> load_u8(const std::string& path, size_t n);
size_t file_size(const std::string& path);
Stim stim_odor(const std::string& o);
Stim stim_alpn(const std::vector<float>& v);

} // namespace fly
