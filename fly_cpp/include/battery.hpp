#pragma once
// battery.hpp — ethology batteries 1:1 with test_ethology.py + arena helpers.
// tmaze / train_fix / buridan / habituate / detour / heatbox / spaced,
// plus fast protocol tests (test_x / test_std / test_sleep).
// RNG note: Python uses np.random.default_rng (PCG64) / random.Random;
// C++ uses mt19937_64 — trajectories differ bit-wise, behavior class identical.
#include "fly.hpp"
#include <string>
#include <utility>
#include <vector>

namespace fly { namespace battery {

float pi_of(float safe, float shock);

// T-maze (full, DoOR). Saves .wbin if w_out non-empty.
void tmaze(const std::string& data, const std::string& w_out = "");

// Visual helpers (64x64, float 0..1)
std::vector<float> bar_img(float u, int width = 5, float bg = 0.0f, float fg = 1.0f);
std::vector<float> render_pano(float x, float y, float th,
                               const std::vector<std::pair<float,float>>& poles,
                               int width = 5);
std::vector<float> disk_img(float cx, float r, float bg = 0.0f, float fg = 1.0f);

// Opponent stripe-fixation outputs (retinotopic pools, seeds 21/22)
FlyBrain make_fix_api(const std::string& data);
void train_fix_on_brain(FlyBrain& api);
void train_fix(const std::string& data, const std::string& w_out = "");

// Pacing Buridan. w_in: .wbin from train_fix (if missing, trains inline).
void buridan(const std::string& data, const std::string& w_in = "",
             int steps = 25, int n_switch = 3);

// Habituation (STD off flat + STD on depress + recovery). Flee-only arena
// outputs trained inline (eat/food are separate scope=new outputs and do
// not affect flee) unless w_in (.wbin) exists.
void habituate(const std::string& data, const std::string& w_in = "");

// Gotz detour re-acquisition.
void detour(const std::string& data, const std::string& w_in = "", int steps = 20);

// Heatbox operant place learning (1D chamber, alpn= prosthesis, 8 bins)
std::pair<std::vector<float>, int> place_code(float x);
std::vector<int> heatbox_run(const std::string& data, const std::string& cond,
                             const std::vector<int>& sched,
                             int steps = 100, int test = 40, float pun_us = 0.5f);
void heatbox(const std::string& data, const std::string& sched_out = "hb_sched.bin",
             int steps = 100, int test = 40);
void heatbox_ctl(const std::string& data, const std::string& sched_in = "hb_sched.bin",
                 int steps = 100, int test = 40);

// Massed vs spaced + SHY retention (no consolidation mechanism)
void spaced(const std::string& data);

// Fast protocol tests (mb, seconds)
void test_x(const std::string& data);
void test_std(const std::string& data);
void test_sleep(const std::string& data);

}} // namespace fly::battery
