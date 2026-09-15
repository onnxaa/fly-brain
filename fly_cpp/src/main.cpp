// main.cpp — fly CLI (twin driver). Phase 1: info/step/train.
// Batteries (tmaze/heatbox/...) land in battery.cpp (phase 2).
#include "fly.hpp"
#include <cstdio>
#include <cstring>
#include <string>

static void usage() {
    std::printf("usage: fly --data DIR --mode mb|full [--seed N] <cmd> [opts]\n"
                "  info | step | train\n");
}

int main(int argc, char** argv) {
    std::string data = ".", mode = "mb", cmd;
    std::string odor;
    float reward = 0, punish = 0;
    int hops = -1;
    float thr = 0;
    std::string act, w_in, w_out, alpn_bin;
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
        else if (a == "--reward") reward = std::stof(argv[++i]);
        else if (a == "--punish") punish = std::stof(argv[++i]);
        else if (a == "--hops") hops = std::stoi(argv[++i]);
        else if (a == "--thr") thr = std::stof(argv[++i]);
        else if (a == "--seed") { ++i; }
        else if (a[0] != '-') cmd = a;
    }
    if (cmd.empty()) { usage(); return 1; }
    fly::FlyBrain b(mode, data);
    b.enable_scaling();
    if (!act.empty()) b.set_activation(act);
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
    if (cmd == "step") {
        fly::Out o = b.step(s, hops, thr);
        std::printf("MB_pref=%+.4f MB_app=%+.4f MB_avo=%+.4f KC_active=%d "
                    "DAN_pam=%+.4f DAN_ppl=%+.4f ALPN_mean=%+.4f EI_sum=%.1f\n",
                    o.MB_pref, o.MB_app, o.MB_avo, o.KC_active, o.DAN_pam, o.DAN_ppl,
                    o.ALPN_mean, o.EI_sum);
        return 0;
    }
    if (cmd == "train") {
        fly::Out o = b.train(s, reward, punish, true, hops, thr);
        if (!w_out.empty()) b.save_wbin(w_out);
        std::printf("MB_pref=%+.4f MB_app=%+.4f MB_avo=%+.4f KC_active=%d\n", o.MB_pref,
                    o.MB_app, o.MB_avo, o.KC_active);
        return 0;
    }
    std::printf("unknown cmd\n");
    return 1;
}
