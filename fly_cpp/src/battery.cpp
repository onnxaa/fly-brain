// battery.cpp — ethology batteries 1:1 with test_ethology.py (+arena flee).
// See battery.hpp for the RNG note (PCG64 vs mt19937_64: same behavior class).
#include "battery.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

namespace fly { namespace battery {

static const int IMG = 64;
static const double PI = 3.14159265358979323846;

float pi_of(float safe, float shock) {
    return (safe - shock) / (std::fabs(safe) + std::fabs(shock) + 1e-9f);
}

static Stim s_odor(const std::string& o) { Stim s; s.has_odor_str = true; s.odor_str = o; return s; }
static Stim s_alpn(const std::vector<float>& v) { Stim s; s.has_alpn = true; s.alpn = v; return s; }
static Stim s_img(const std::vector<float>& im) {
    Stim s; s.has_image = true; s.image = im; s.imgH = IMG; s.imgW = IMG; return s;
}
static bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
}
static float getX(const Out& o, const std::string& n) {
    auto it = o.X.find(n);
    return it == o.X.end() ? 0.0f : it->second;
}

// ---- tmaze ----
void tmaze(const std::string& data, const std::string& w_out) {
    const std::string CS_SHOCK = "methyl_salicylate", CS_SAFE = "ethyl_hexanoate";
    FlyBrain api("full", data);
    api.enable_scaling();
    float pre_s = api.step(s_odor(CS_SAFE)).MB_pref;
    float pre_p = api.step(s_odor(CS_SHOCK)).MB_pref;
    std::printf("naive: safe=%+.2f shock=%+.2f PI=%+.2f\n", pre_s, pre_p, pi_of(pre_s, pre_p));
    api.train(s_odor(CS_SHOCK), 0.0f, 1.0f);
    float s1 = api.step(s_odor(CS_SAFE)).MB_pref;
    float p1 = api.step(s_odor(CS_SHOCK)).MB_pref;
    std::printf("1-trial: safe=%+.2f shock=%+.2f PI=%+.2f (expect >0)\n", s1, p1, pi_of(s1, p1));
    for (int i = 0; i < 4; i++) api.train(s_odor(CS_SHOCK), 0.0f, 1.0f);
    float s3 = api.step(s_odor(CS_SAFE)).MB_pref;
    float p3 = api.step(s_odor(CS_SHOCK)).MB_pref;
    std::printf("5-trial: safe=%+.2f shock=%+.2f PI=%+.2f (expect >0)\n", s3, p3, pi_of(s3, p3));
    for (int i = 0; i < 3; i++) {
        api.train(s_odor(CS_SAFE), 0.0f, 1.0f);
        api.train(s_odor(CS_SHOCK), 1.0f, 0.0f);
    }
    float sr = api.step(s_odor(CS_SAFE)).MB_pref;
    float pr = api.step(s_odor(CS_SHOCK)).MB_pref;
    std::printf("reversed: safe=%+.2f shock=%+.2f PI=%+.2f (expect <0)\n", sr, pr, pi_of(sr, pr));
    if (!w_out.empty()) { api.save_wbin(w_out); std::printf("saved %s\n", w_out.c_str()); }
}

// ---- visual helpers ----
std::vector<float> bar_img(float u, int width, float bg, float fg) {
    std::vector<float> img((size_t)IMG * IMG, bg);
    int c = (int)std::lround(32 * (1 + u));
    int h = width / 2;
    for (int y = 0; y < IMG; y++)
        for (int x = std::max(0, c - h); x <= std::min(IMG - 1, c + h); x++)
            img[(size_t)y * IMG + x] = fg;
    return img;
}

std::vector<float> render_pano(float x, float y, float th,
                               const std::vector<std::pair<float,float>>& poles,
                               int width) {
    std::vector<float> img((size_t)IMG * IMG, 0.0f);
    float ax = std::sin(th), ay = std::cos(th);
    float nx = std::cos(th), ny = -std::sin(th);
    int h = width / 2;
    for (auto& pl : poles) {
        float dx = pl.first - x, dy = pl.second - y;
        float fwd = dx * ax + dy * ay, lat = dx * nx + dy * ny;
        if (fwd <= 0.5f) continue;
        float u = std::atan2(lat, fwd) / (float)(PI / 2);
        if (std::fabs(u) > 1) continue;
        int c = (int)std::lround(32 * (1 + u));
        for (int yy = 0; yy < IMG; yy++)
            for (int xx = std::max(0, c - h); xx <= std::min(IMG - 1, c + h); xx++)
                img[(size_t)yy * IMG + xx] = 1.0f;
    }
    return img;
}

std::vector<float> disk_img(float cx, float r, float bg, float fg) {
    std::vector<float> img((size_t)IMG * IMG, bg);
    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++) {
            float dx = (float)x - 32 - cx, dy = (float)y - 32;
            if (dx * dx + dy * dy <= r * r) img[(size_t)y * IMG + x] = fg;
        }
    return img;
}

static float img_sum(const std::vector<float>& im) {
    double s = 0; for (auto v : im) s += v; return (float)s;
}

// ---- fixation outputs ----
FlyBrain make_fix_api(const std::string& data) {
    FlyBrain api("full", data);
    api.enable_scaling();
    std::vector<int32_t> pL, pR;
    for (size_t i = 0; i < api.R.size(); i++) {
        if (api.R_cx[i] < 0.4f) pL.push_back(api.R[i]);
        if (api.R_cx[i] > 0.6f) pR.push_back(api.R[i]);
    }
    api.x_add_output("fixL", 8, pL, -1, 0.01f, 21);
    api.x_add_output("fixR", 8, pR, -1, 0.01f, 22);
    return api;
}

void train_fix_on_brain(FlyBrain& api) {
    Out o = api.step(s_img(bar_img(-0.5f)));
    std::printf("naive u=-0.5: steer=%+.3f (expect <0 from retinotopy alone)\n",
                getX(o, "fixR") - getX(o, "fixL"));
    const float UU[7] = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    for (int t = 0; t < 3; t++) {
        for (float u : UU) {
            std::vector<float> im = bar_img(u);
            Stim st = s_img(im);
            api.x_teach_output("fixL", 1, 1.2f, true, true, std::max(0.0f, -u),
                               "new", 0.02f, st);
            api.x_teach_output("fixR", 1, 1.2f, true, true, std::max(0.0f, u),
                               "new", 0.02f, st);
        }
        std::printf(" round %d/3\n", t + 1);
    }
    const float EV[5] = {-0.5f, -0.125f, 0.0f, 0.125f, 0.5f};
    for (float u : EV) {
        Out oo = api.step(s_img(bar_img(u)));
        std::printf("u=%+.2f fixL=%.3f fixR=%.3f steer=%+.3f\n", u,
                    getX(oo, "fixL"), getX(oo, "fixR"),
                    getX(oo, "fixR") - getX(oo, "fixL"));
    }
}

void train_fix(const std::string& data, const std::string& w_out) {
    FlyBrain api = make_fix_api(data);
    train_fix_on_brain(api);
    if (!w_out.empty()) { api.save_wbin(w_out); std::printf("saved %s\n", w_out.c_str()); }
}

static float ang_dist(float a, float b) {
    float d = std::fmod(a - b + (float)PI, 2 * (float)PI);
    if (d < 0) d += 2 * (float)PI;
    return std::fabs(d - (float)PI);
}
static float pole_bearing(float x, float y, float th, std::pair<float,float> pole) {
    float dx = pole.first - x, dy = pole.second - y;
    return ang_dist(th, std::atan2(dx, dy));
}
static std::string lags_str(const std::vector<int>& lags) {
    std::string s;
    for (size_t i = 0; i < lags.size(); i++) {
        if (i) s += ",";
        s += (lags[i] < 0 ? std::string("None") : std::to_string(lags[i]));
    }
    return s;
}

void buridan(const std::string& data, const std::string& w_in, int steps, int n_switch) {
    FlyBrain api = make_fix_api(data);
    bool loaded = false;
    if (!w_in.empty() && file_exists(w_in)) {
        try { api.load_wbin(w_in); loaded = true; } catch (...) {}
    }
    if (!loaded) train_fix_on_brain(api);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> gust(-0.12f, 0.12f);
    const float DEG25 = 25.0f * (float)PI / 180.0f;
    for (int steer_on = 1; steer_on >= 0; steer_on--) {
        float x = 0.0f, y = -4.0f, th = 0.0f;
        int face = 0, tot = 0;
        std::vector<int> lags;
        for (int sw = 0; sw < n_switch; sw++) {
            std::pair<float,float> pole = (sw % 2 == 0) ?
                std::make_pair(-4.0f, 8.0f) : std::make_pair(4.0f, 8.0f);
            int lag = -2; // -2 = waiting, -1 = None, >=0 found
            for (int i = 0; i < steps; i++) {
                th += gust(rng);
                std::vector<std::pair<float,float>> ps = {pole};
                std::vector<float> im = render_pano(x, y, th, ps);
                if (img_sum(im) == 0) th += 0.15f;
                else if (steer_on) {
                    Out oo = api.step(s_img(im));
                    float s = getX(oo, "fixR") - getX(oo, "fixL");
                    if (std::fabs(s) < 0.04f) s = 0.0f;
                    float d = 0.8f * s;
                    if (d > 0.5f) d = 0.5f; if (d < -0.5f) d = -0.5f;
                    th += d;
                }
                x += 0.35f * std::sin(th); y += 0.35f * std::cos(th);
                if (x > 9) x = 9; if (x < -9) x = -9;
                if (y > 9) y = 9; if (y < -9) y = -9;
                float e = pole_bearing(x, y, th, pole);
                tot++; if (e < DEG25) face++;
                if (lag == -2) {
                    if (e < DEG25) { lags.push_back(i + 1); lag = i; }
                    else if (i == steps - 1) lags.push_back(-1);
                }
            }
        }
        std::printf("%s: facing=%.0f%% reface-lags=[%s]\n",
                    steer_on ? "STEER" : "BASE",
                    tot ? face * 100.0 / tot : 0.0, lags_str(lags).c_str());
    }
}

// ---- habituate (flee-only arena) ----
static FlyBrain make_flee_api(const std::string& data) {
    FlyBrain api("full", data);
    api.enable_scaling();
    std::vector<int32_t> pool;
    pool.insert(pool.end(), api.R.begin(), api.R.end());
    pool.insert(pool.end(), api.MEv.begin(), api.MEv.end());
    pool.insert(pool.end(), api.LOv.begin(), api.LOv.end());
    api.x_add_output("fleeL", 8, pool, -1, 0.01f, 32);
    api.x_add_output("fleeR", 8, pool, -1, 0.01f, 33);
    return api;
}
static void train_flee_on_brain(FlyBrain& api) {
    const float sizes[3] = {6, 14, 28};
    for (int t = 0; t < 2; t++) {
        for (float r : sizes) {
            api.x_teach_output("fleeL", 1, 1.2f, true, true, r / 28.0f,
                               "new", 0.02f, s_img(disk_img(-16, r)));
            api.x_teach_output("fleeR", 1, 1.2f, true, true, r / 28.0f,
                               "new", 0.02f, s_img(disk_img(16, r)));
            api.x_teach_output("fleeL", 1, 1.2f, true, true, r / 28.0f,
                               "new", 0.02f, s_img(disk_img(0, r)));
            api.x_teach_output("fleeR", 1, 1.2f, true, true, r / 28.0f,
                               "new", 0.02f, s_img(disk_img(0, r)));
        }
        std::printf(" flee round %d/2\n", t + 1);
    }
}

void habituate(const std::string& data, const std::string& w_in) {
    FlyBrain api = make_flee_api(data);
    bool loaded = false;
    if (!w_in.empty() && file_exists(w_in)) {
        try { api.load_wbin(w_in); loaded = true; } catch (...) {}
    }
    if (!loaded) train_flee_on_brain(api);
    auto flee = [&](const Out& o) { return std::max(getX(o, "fleeL"), getX(o, "fleeR")); };
    std::printf("-- STD off (default): stability control\n");
    std::vector<float> seq;
    for (int i = 0; i < 5; i++) seq.push_back(flee(api.step(s_img(disk_img(0, 20)))));
    std::printf("5x same loom: %.3f..%.3f (flat: no spurious depression, escape stays reliable)\n",
                seq.front(), seq.back());
    for (float r : {4.0f, 10.0f, 20.0f, 28.0f})
        std::printf("size r=%.0f: flee=%.3f\n", r, flee(api.step(s_img(disk_img(0, r)))));
    std::printf("-- STD on (protocol mode, alpha=0.1 tau=25): habituation\n");
    api.set_std(0.1f, 25.0f);
    seq.clear();
    for (int i = 0; i < 15; i++) seq.push_back(flee(api.step(s_img(disk_img(0, 20)))));
    std::printf("loom t1=%.2f t5=%.2f t10=%.2f t15=%.2f drop=%.0f%%\n",
                seq[0], seq[4], seq[9], seq[14],
                (1 - seq[14] / std::max(seq[0], 1e-9f)) * 100);
    float sh = flee(api.step(s_img(disk_img(16, 20))));
    std::printf("shifted bearing: %.2f (specificity gradient: habituated %.2f < shifted < naive)\n",
                sh, seq[14]);
    std::vector<float> blank((size_t)IMG * IMG, 0.0f);
    for (int i = 0; i < 40; i++) api.step(s_img(blank));
    float rec = flee(api.step(s_img(disk_img(0, 20))));
    std::printf("after 40 quiet trials: %.2f (recovery vs t1 %.2f)\n", rec, seq[0]);
    api.set_std(0.0f, 25.0f);
    std::printf("STD back off\n");
}

void detour(const std::string& data, const std::string& w_in, int steps) {
    FlyBrain api = make_fix_api(data);
    bool loaded = false;
    if (!w_in.empty() && file_exists(w_in)) {
        try { api.load_wbin(w_in); loaded = true; } catch (...) {}
    }
    if (!loaded) train_fix_on_brain(api);
    float x = 0.0f, y = -4.0f, th = 0.0f;
    std::pair<float,float> tgt = {4.0f, 8.0f};
    int ret = -1;
    std::map<std::string, float> errs;
    for (int i = 0; i < 3 * steps; i++) {
        std::vector<std::pair<float,float>> poles;
        std::string phase;
        if (i < steps) { poles = {tgt}; phase = "target"; }
        else if (i < 2 * steps) { poles = {{-4.0f, 8.0f}}; phase = "distractor"; }
        else { poles = {tgt}; phase = "return"; }
        std::vector<float> im = render_pano(x, y, th, poles);
        if (img_sum(im) == 0) th += 0.15f;
        else {
            Out oo = api.step(s_img(im));
            float s = getX(oo, "fixR") - getX(oo, "fixL");
            if (std::fabs(s) < 0.04f) s = 0.0f;
            float d = 0.8f * s;
            if (d > 0.5f) d = 0.5f; if (d < -0.5f) d = -0.5f;
            th += d;
        }
        x += 0.35f * std::sin(th); y += 0.35f * std::cos(th);
        if (x > 9) x = 9; if (x < -9) x = -9;
        if (y > 9) y = 9; if (y < -9) y = -9;
        float e = pole_bearing(x, y, th, tgt);
        if (i == steps - 1 || i == 2 * steps - 1 || i == 3 * steps - 1) {
            errs[phase] = e;
            std::printf("end-%s: heading-err=%.0fdeg\n", phase.c_str(), e * 180.0f / (float)PI);
        }
        const float DEG25 = 25.0f * (float)PI / 180.0f;
        if (phase == "return" && ret < 0 && e < DEG25) ret = i - 2 * steps + 1;
    }
    float e_end = pole_bearing(x, y, th, tgt) * 180.0f / (float)PI;
    bool cap = errs["distractor"] * 180.0f / (float)PI > 45;
    std::printf("captured-by-distractor=%s return-time=%d/%d final-err=%.0fdeg\n",
                cap ? "yes" : "no", ret, steps, e_end);
}

// ---- heatbox ----
std::pair<std::vector<float>, int> place_code(float x) {
    const int B = 8;
    int b = (int)((x + 10) / 20 * B);
    if (b < 0) b = 0; if (b > B - 1) b = B - 1;
    std::vector<float> v((size_t)B, 0.0f);
    v[(size_t)b] = 1.0f;
    return {v, b};
}

std::vector<int> heatbox_run(const std::string& data, const std::string& cond,
                             const std::vector<int>& sched,
                             int steps, int test, float pun_us) {
    FlyBrain api("full", data);
    api.enable_scaling();
    float init[8];
    for (int b = 0; b < 8; b++) {
        float xc = -10 + 20 * (b + 0.5f) / 8;
        init[b] = s_alpn(place_code(xc).first).has_alpn ?
            api.step(s_alpn(place_code(xc).first)).MB_pref : 0.0f;
    }
    std::vector<int> usched = sched;
    if (cond == "unpaired" && !sched.empty()) {
        usched = sched;
        std::mt19937_64 rng(3);
        for (size_t i = usched.size(); i > 1; i--) {
            std::uniform_int_distribution<size_t> d(0, i - 1);
            std::swap(usched[i - 1], usched[d(rng)]);
        }
    }
    float x = (cond == "yoked") ? 5.0f : -5.0f;
    int direction = (cond == "yoked") ? -1 : 1;
    int hot1 = 0, hot2 = 0, cool = 0;
    std::vector<int> sched_out;
    for (int i = 0; i < steps; i++) {
        auto pc = place_code(x);
        float pref = api.step(s_alpn(pc.first)).MB_pref;
        if (cool > 0) cool--;
        else if (pref < init[pc.second] - 2.0f) { direction *= -1; cool = 6; }
        x += direction * 0.5f;
        if (x > 10) { x = 10.0f; direction = -1; }
        if (x < -10) { x = -10.0f; direction = 1; }
        bool hot = x > 0;
        bool pun = false;
        if (cond == "contingent") pun = hot;
        else if (cond == "naive") pun = false;
        else pun = (i < (int)usched.size()) ? (usched[(size_t)i] != 0) : false;
        if (pun) api.train(s_alpn(pc.first), 0.0f, pun_us);
        sched_out.push_back(pun ? 1 : 0);
        if (hot) { if (i < steps / 2) hot1++; else hot2++; }
    }
    int thot = 0;
    for (int i = 0; i < test; i++) {
        auto pc = place_code(x);
        float pref = api.step(s_alpn(pc.first)).MB_pref;
        if (cool > 0) cool--;
        else if (pref < init[pc.second] - 2.0f) { direction *= -1; cool = 6; }
        x += direction * 0.5f;
        if (x > 10) { x = 10.0f; direction = -1; }
        if (x < -10) { x = -10.0f; direction = 1; }
        if (x > 0) thot++;
    }
    int h = steps / 2;
    std::printf("%s: train-hot%% %.0f/%.0f test-hot%%=%.0f\n", cond.c_str(),
                hot1 * 100.0 / std::max(h, 1), hot2 * 100.0 / std::max(h, 1),
                thot * 100.0 / std::max(test, 1));
    return sched_out;
}

void heatbox(const std::string& data, const std::string& sched_out, int steps, int test) {
    std::vector<int> s = heatbox_run(data, "contingent", {}, steps, test);
    FILE* f = std::fopen(sched_out.c_str(), "wb");
    if (f) {
        for (int v : s) { int8_t b = v ? 1 : 0; std::fwrite(&b, 1, 1, f); }
        std::fclose(f);
    }
    heatbox_run(data, "naive", {}, steps, test);
}

void heatbox_ctl(const std::string& data, const std::string& sched_in, int steps, int test) {
    std::vector<int> s;
    FILE* f = std::fopen(sched_in.c_str(), "rb");
    if (f) {
        int c; while ((c = std::fgetc(f)) != EOF) s.push_back(c ? 1 : 0);
        std::fclose(f);
    }
    heatbox_run(data, "yoked", s, steps, test);
    heatbox_run(data, "unpaired", s, steps, test);
}

// ---- spaced ----
static int sleep_bout(FlyBrain& api, int want_asleep, int max_steps = 120) {
    int slept = 0;
    Stim cs = s_odor("ethyl_hexanoate");
    for (int i = 0; i < max_steps; i++) {
        Out o = api.step(cs);
        if (o.asleep) { if (++slept >= want_asleep) break; }
    }
    for (int i = 0; i < 40; i++) {
        Out o = api.step(cs);
        if (!o.asleep) break;
    }
    return slept;
}
static void spaced_setup(FlyBrain& api) {
    api.enable_clock();
    api.enable_auto_sleep(1.0f);
    api.tick_clock(12, 0.0f);
}
static float spaced_pi(FlyBrain& api) {
    float s = api.step(s_odor("ethyl_hexanoate")).MB_pref;
    float p = api.step(s_odor("methyl_salicylate")).MB_pref;
    return pi_of(s, p);
}

void spaced(const std::string& data) {
    FlyBrain m("full", data); m.enable_scaling();
    for (int i = 0; i < 5; i++) m.train(s_odor("methyl_salicylate"), 0.0f, 1.0f);
    std::printf("massed immediate PI=%+.2f\n", spaced_pi(m));
    FlyBrain s("full", data); s.enable_scaling();
    spaced_setup(s);
    for (int i = 0; i < 5; i++) {
        s.train(s_odor("methyl_salicylate"), 0.0f, 1.0f);
        s.tick_clock(2, 0.0f);
        sleep_bout(s, 4);
    }
    std::printf("spaced immediate PI=%+.2f (expect ~= massed: no interference mechanism)\n",
                spaced_pi(s));
    spaced_setup(m);
    m.tick_clock(12, 0.0f);
    int n1 = sleep_bout(m, 40);
    std::printf("massed retention PI=%+.2f after 24h + %d sleep steps (SHY wash = the only forgetting)\n",
                spaced_pi(m), n1);
    s.tick_clock(12, 0.0f);
    int n2 = sleep_bout(s, 40);
    std::printf("spaced retention PI=%+.2f after 24h + %d sleep steps (no consolidation: no LTM/ARM split)\n",
                spaced_pi(s), n2);
    FlyBrain a("full", data); a.enable_scaling();
    for (int i = 0; i < 5; i++) a.train(s_odor("methyl_salicylate"), 0.0f, 1.0f);
    float p0 = spaced_pi(a), p1 = spaced_pi(a);
    std::printf("awake control: PI=%+.2f..%+.2f (nothing decays awake)\n", p0, p1);
}

// ---- fast protocol tests ----
void test_x(const std::string& data) {
    FlyBrain b("mb", data);
    b.enable_scaling();
    b.x_add_output("t", 8, {}, -1, -1, 0);
    for (int i = 0; i < 3; i++) {
        Stim sa = s_odor("A"), sb = s_odor("B");
        b.x_teach_output("t", 1, 0.3f, true, false, 0.0f, "new", 0.02f, sa);
        b.x_teach_output("t", 1, 0.3f, false, false, 0.0f, "new", 0.02f, sb);
    }
    float a = getX(b.step(s_odor("A")), "t");
    float c = getX(b.step(s_odor("B")), "t");
    auto rep = b.x_report();
    std::printf("X_t(A)=%.3f X_t(B)=%.3f sep=%.3f drift=%.4f%%\n",
                a, c, a - c, rep["EI_drift_%"]);
    std::printf("%s\n", (a > c && rep["EI_drift_%"] < 1.0) ? "PASS-x" : "FAIL-x");
}

void test_std(const std::string& data) {
    FlyBrain b("mb", data);
    b.enable_scaling();
    Stim sa = s_odor("A");
    std::vector<float> off;
    for (int i = 0; i < 5; i++) off.push_back(b.step(sa).MB_app);
    bool flat = (off.front() - off.back()) == 0 ||
                std::fabs(off.front() - off.back()) / std::max(std::fabs(off.front()), 1e-9f) < 1e-6;
    std::printf("STD off 5x: %.3f..%.3f flat=%s\n", off.front(), off.back(), flat ? "yes" : "no");
    b.set_std(0.1f, 25.0f);
    std::vector<float> seq;
    for (int i = 0; i < 15; i++) seq.push_back(b.step(sa).MB_app);
    float drop = (1 - seq[14] / std::max(seq[0], 1e-9f)) * 100;
    std::printf("STD on t1=%.2f t5=%.2f t15=%.2f drop=%.0f%%\n", seq[0], seq[4], seq[14], drop);
    Stim blank; blank.has_alpn = true; blank.alpn = std::vector<float>(8, 0.0f);
    for (int i = 0; i < 40; i++) b.step(blank);
    float rec = b.step(sa).MB_app;
    std::printf("after 40 quiet: %.2f vs t1 %.2f\n", rec, seq[0]);
    b.set_std(0.0f, 25.0f);
    std::printf("%s\n", (flat && drop > 30 && rec > seq[14]) ? "PASS-std" : "FAIL-std");
}

void test_sleep(const std::string& data) {    FlyBrain b("mb", data);
    b.enable_scaling();
    Stim sa = s_odor("A");
    float naive = b.step(sa).MB_pref;
    b.train(sa, 1.0f, 0.0f);
    float tr = b.step(sa).MB_pref;
    b.sleep(10, 0.02f);
    float sl = b.step(sa).MB_pref;
    std::printf("naive=%+.2f trained=%+.2f sleep=%+.2f\n", naive, tr, sl);
    bool toward = (std::fabs(sl - naive) < std::fabs(tr - naive));
    std::printf("%s (SHY wash toward baseline)\n", toward ? "PASS-sleep" : "FAIL-sleep");
}

void bench(const std::string& data, const std::string& mode, int steps) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    auto t0 = clk::now();
    FlyBrain b(mode, data);
    auto t1 = clk::now();
    b.enable_scaling();
    auto t2 = clk::now();
    Stim s;
    if (mode == "full") { s.has_odor_str = true; s.odor_str = "geosmin"; }
    else { s.has_odor_str = true; s.odor_str = "A"; }
    // warmup (page in, thread pool spin-up)
    b.step(s);
    auto t3 = clk::now();
    for (int i = 0; i < steps; i++) b.step(s);
    auto t4 = clk::now();
    for (int i = 0; i < 2; i++) b.train(s, 0.0f, 1.0f);
    auto t5 = clk::now();
    b.sleep(5, 0.02f);
    auto t6 = clk::now();
    std::printf("bench mode=%s N=%d E=%lld\n", mode.c_str(), b.N, (long long)b.E);
    std::printf("  load=%lldms scaling=%lldms step1=%lldms step_avg=%lldms train_avg=%lldms sleep5=%lldms\n",
                (long long)ms(t0, t1), (long long)ms(t1, t2), (long long)ms(t2, t3),
                (long long)ms(t3, t4) / steps, (long long)ms(t4, t5) / 2,
                (long long)ms(t5, t6));
}

}} // namespace fly::battery
