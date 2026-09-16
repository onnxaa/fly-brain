// main.cpp — fly CLI (twin driver). Phase 1: info/step/train.
// Phase 2: batteries (tmaze/train_fix/buridan/habituate/detour/
// heatbox/heatbox_ctl/spaced) + fast protocol tests (test-x/std/sleep).
#include "battery.hpp"
#include "fly.hpp"
#include <cstdio>
#include <cstring>
#include <string>

static void usage() {
    std::printf("usage: fly --data DIR --mode mb|full|banc|mcns [--seed N] [--vnc] <cmd> [opts]\n"
                "  info | step | train\n"
                "  tmaze [--w-out F] | train_fix [--w-out F]\n"
                "  buridan [--w-in F] [--steps N] [--n-switch K]\n"
                "  habituate [--w-in F] | detour [--w-in F] [--steps N]\n"
                "  heatbox [--sched F] [--steps N] [--test N]\n"
                "  heatbox_ctl [--sched F] [--steps N] [--test N] | spaced\n"
                "  test-x | test-std | test-sleep | bench [--steps N]\n");
}

int main(int argc, char** argv) {
    std::string data = ".", mode = "mb", cmd;
    std::string odor;
    float reward = 0, punish = 0;
    int hops = -1;
    float thr = 0;
    std::string act, w_in, w_out, alpn_bin, sched = "";
    std::string mech_bin;
    bool use_vnc = false;
    int steps = -1, test_n = -1, n_switch = -1;
    std::string scaling, img_bin; int imgH = 0, imgW = 0, vpol = 0;
    bool stdp = false;
    float leak = 0.0f;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](std::string& dst) { dst = argv[++i]; };
        if (a == "--data") need(data);
        else if (a == "--mode") need(mode);
        else if (a == "--odor") need(odor);
        else if (a == "--act") need(act);
        else if (a == "--w-in") need(w_in);
        else if (a == "--w-out") need(w_out);
        else if (a == "--alpn-bin") need(alpn_bin);
        else if (a == "--mech-bin") need(mech_bin);
        else if (a == "--vnc") use_vnc = true;
        else if (a == "--sched") need(sched);
        else if (a == "--reward") reward = std::stof(argv[++i]);
        else if (a == "--punish") punish = std::stof(argv[++i]);
        else if (a == "--hops") hops = std::stoi(argv[++i]);
        else if (a == "--thr") thr = std::stof(argv[++i]);
        else if (a == "--steps") steps = std::stoi(argv[++i]);
        else if (a == "--test") test_n = std::stoi(argv[++i]);
        else if (a == "--n-switch") n_switch = std::stoi(argv[++i]);
        else if (a == "--seed") { ++i; }
        else if (a == "--scaling") need(scaling);
        else if (a == "--img-bin") need(img_bin);
        else if (a == "--img-h") imgH = std::stoi(argv[++i]);
        else if (a == "--img-w") imgW = std::stoi(argv[++i]);
        else if (a == "--vpol") vpol = std::stoi(argv[++i]);
        else if (a == "--stdp") stdp = true;
        else if (a == "--leak") leak = std::stof(argv[++i]);
        else if (a[0] != '-') cmd = a;
    }
    if (cmd.empty()) { usage(); return 1; }
    // batteries (full mode forced inside, --mode ignored except tests)
    if (cmd == "tmaze") { fly::battery::tmaze(data, w_out, mode); return 0; }
    if (cmd == "train_fix") { fly::battery::train_fix(data, w_out, mode); return 0; }
    if (cmd == "buridan") {
        fly::battery::buridan(data, w_in,
                              steps > 0 ? steps : 25,
                              n_switch > 0 ? n_switch : 3, mode);
        return 0;
    }
    if (cmd == "habituate") { fly::battery::habituate(data, w_in, mode); return 0; }
    if (cmd == "detour") {
        fly::battery::detour(data, w_in, steps > 0 ? steps : 20, mode);
        return 0;
    }
    if (cmd == "heatbox") {
        fly::battery::heatbox(data, sched.empty() ? "hb_sched.bin" : sched,
                              steps > 0 ? steps : 100,
                              test_n > 0 ? test_n : 40, mode);
        return 0;
    }
    if (cmd == "heatbox_ctl") {
        fly::battery::heatbox_ctl(data, sched.empty() ? "hb_sched.bin" : sched,
                                  steps > 0 ? steps : 100,
                                  test_n > 0 ? test_n : 40, mode);
        return 0;
    }
    if (cmd == "spaced") { fly::battery::spaced(data, mode); return 0; }
    if (cmd == "test-x") { fly::battery::test_x(data, mode); return 0; }
    if (cmd == "test-std") { fly::battery::test_std(data, mode); return 0; }
    if (cmd == "test-sleep") { fly::battery::test_sleep(data, mode); return 0; }
    if (cmd == "test-state") { fly::battery::test_state(data, mode); return 0; }
    if (cmd == "test-attractor") { fly::battery::test_attractor(data, mode); return 0; }
    if (cmd == "bench") {
        fly::battery::bench(data, mode, steps > 0 ? steps : 5);
        return 0;
    }

    fly::FlyBrain b(mode, data);
    b.enable_scaling();
    if (use_vnc) b.enable_vnc();
    if (!act.empty()) b.set_activation(act);
    if (!scaling.empty()) b.set_scaling(scaling);
    if (leak > 0) b.set_state(leak);
    if (!w_in.empty()) b.load_wbin(w_in);
    if (cmd == "info") {
        std::printf("mode=%s N=%d E=%lld EI=%.1f\n", mode.c_str(), b.N, (long long)b.E,
                    b.EI_sum());
        return 0;
    }
    fly::Stim s;
    if (!odor.empty()) { s.has_odor_str = true; s.odor_str = odor; }
    if (!alpn_bin.empty()) {
        size_t n = fly::file_size(alpn_bin) / 4;
        s.has_alpn = true;
        s.alpn = fly::load_f32(alpn_bin, n);
    }
    if (!mech_bin.empty()) {
        size_t n = fly::file_size(mech_bin) / 4;
        s.has_mech = true;
        s.mech = fly::load_f32(mech_bin, n);
    }
    if (!img_bin.empty()) {
        size_t n = fly::file_size(img_bin) / 4;
        s.has_image = true;
        s.image = fly::load_f32(img_bin, n);
        s.imgH = imgH > 0 ? imgH : 64; s.imgW = imgW > 0 ? imgW : 64;
        s.vpol = vpol;
    }
    if (cmd == "step") {
        fly::Out o = b.step(s, hops, thr);
        std::printf("MB_pref=%+.4f MB_app=%+.4f MB_avo=%+.4f KC_active=%d "
                    "DAN_pam=%+.4f DAN_ppl=%+.4f ALPN_mean=%+.4f EI_sum=%.1f\n",
                    o.MB_pref, o.MB_app, o.MB_avo, o.KC_active, o.DAN_pam, o.DAN_ppl,
                    o.ALPN_mean, o.EI_sum);
        if (o.has_vnc)
            std::printf("VNC_desc=%+.5f VNC_motor=%+.5f legL=%+.5f legR=%+.5f imb=%+.5f "
                        "wing=%+.5f neck=%+.5f\n",
                        o.VNC_desc_mean, o.VNC_motor, o.VNC_leg_L, o.VNC_leg_R,
                        o.VNC_leg_imb, o.VNC_wing, o.VNC_neck);
        if (o.has_cx)
            std::printf("CX_bump=%d CX_EPG=%+.4f PFL_L=%+.4f PFL_R=%+.4f turn=%+.4f\n",
                        o.CX_bump, o.CX_EPG, o.CX_PFL_L, o.CX_PFL_R, o.CX_turn);
        if (o.has_bcmotor)
            std::printf("BANC_motor=%+.5f legL=%+.5f legR=%+.5f imb=%+.5f "
                        "wing=%+.5f neck=%+.5f\n",
                        o.BANC_motor, o.BANC_leg_L, o.BANC_leg_R,
                        o.BANC_leg_imb, o.BANC_wing, o.BANC_neck);
        return 0;
    }
    if (cmd == "train") {
        fly::Out o = b.train(s, reward, punish, true, hops, thr, stdp);
        if (!w_out.empty()) b.save_wbin(w_out);
        std::printf("MB_pref=%+.4f MB_app=%+.4f MB_avo=%+.4f KC_active=%d\n", o.MB_pref,
                    o.MB_app, o.MB_avo, o.KC_active);
        return 0;
    }
    std::printf("unknown cmd\n");
    return 1;
}
