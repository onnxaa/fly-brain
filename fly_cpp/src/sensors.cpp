// sensors.cpp — vision front end 1:1 with fly_api.py
// (_gray / _sample_R / _small16 / _motion_energies).
#include "fly.hpp"
#include <algorithm>
#include <cmath>

namespace fly {

std::vector<float> FlyBrain::gray(const std::vector<float>& img, int H, int Wd) const {
    std::vector<float> g(H * Wd);
    for (int i = 0; i < H * Wd; i++) {
        float v = img[i];
        g[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    return g;
}

std::vector<float> FlyBrain::sample_R(const std::vector<float>& g, int H, int Wd) const {
    // 3x3 Gaussian RF [[1,2,1],[2,4,2],[1,2,1]]/16, edge-replicated pad
    std::vector<float> b(H * Wd);
    auto at = [&](int y, int x) -> float {
        y = y < 0 ? 0 : (y >= H ? H - 1 : y);
        x = x < 0 ? 0 : (x >= Wd ? Wd - 1 : x);
        return g[(size_t)y * Wd + x];
    };
    for (int y = 0; y < H; y++)
        for (int x = 0; x < Wd; x++)
            b[(size_t)y * Wd + x] =
                (4 * at(y, x) + 2 * (at(y - 1, x) + at(y + 1, x) + at(y, x - 1) + at(y, x + 1)) +
                 at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1)) / 16.0f;
    std::vector<float> o(R.size());
    for (size_t i = 0; i < R.size(); i++) {
        double xs = (double)R_cx[i] * (Wd - 1), ys = (double)R_cy[i] * (H - 1);
        int64_t x0 = (int64_t)std::floor(xs), y0 = (int64_t)std::floor(ys);
        x0 = std::max<int64_t>(0, std::min<int64_t>(x0, Wd - 1));
        y0 = std::max<int64_t>(0, std::min<int64_t>(y0, H - 1));
        int64_t x1 = std::min<int64_t>(x0 + 1, Wd - 1), y1 = std::min<int64_t>(y0 + 1, H - 1);
        float fx = (float)(xs - x0), fy = (float)(ys - y0);
        float Ia = b[(size_t)y0 * Wd + x0], Ib = b[(size_t)y0 * Wd + x1];
        float Ic = b[(size_t)y1 * Wd + x0], Id = b[(size_t)y1 * Wd + x1];
        float v = Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy) + Ic * (1 - fx) * fy + Id * fx * fy;
        o[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    return o;
}

std::vector<float> FlyBrain::small16(const std::vector<float>& g, int H, int Wd) const {
    int bh = std::max(1, H / 16), bw = std::max(1, Wd / 16);
    std::vector<float> s(256, 0.0f);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            double acc = 0;
            for (int a = 0; a < bh; a++)
                for (int b2 = 0; b2 < bw; b2++)
                    acc += g[(size_t)(i * bh + a) * Wd + (j * bw + b2)];
            s[(size_t)i * 16 + j] = (float)(acc / (bh * bw));
        }
    return s;
}

std::pair<std::map<std::string, float>, bool> FlyBrain::motion_energies(
    const std::vector<float>& small) {
    std::map<std::string, float> out{{"R", 0}, {"L", 0}, {"U", 0}, {"D", 0}};
    if (!has_last_small) {
        last_small = small;
        has_last_small = true;
        return {out, true};
    }
    const std::vector<float>& p = last_small;
    auto roll_sum = [&](int dy, int dx, int dy2, int dx2) {
        double acc = 0;
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                float s = small[(size_t)y * 16 + x];
                float p1 = p[(size_t)((y + dy + 16) % 16) * 16 + ((x + dx + 16) % 16)];
                float p2 = p[(size_t)((y + dy2 + 16) % 16) * 16 + ((x + dx2 + 16) % 16)];
                acc += std::max(0.0f, s * p1 - s * p2);
            }
        return (float)acc;
    };
    // Python: R = max(0, s*roll(p,+1,axis=1) - s*roll(p,-1,axis=1))
    // np.roll(p,+1,axis=1)[y,x] = p[y,x-1]
    out["R"] = roll_sum(0, -1, 0, +1);
    out["L"] = roll_sum(0, +1, 0, -1);
    out["D"] = roll_sum(-1, 0, +1, 0);
    out["U"] = roll_sum(+1, 0, -1, 0);
    last_small = small;
    return {out, false};
}

} // namespace fly
