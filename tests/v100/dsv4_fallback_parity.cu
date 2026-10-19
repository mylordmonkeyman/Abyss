// Model-free P0 checks of existing DeepSeek kernels against host scalar references.
// No candidate kernels, quantization changes, model weights, or timing claims.
#include "strata/kernels/dsv4_fast.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace ds = strata::kernels::dsv4;
namespace {
constexpr int H = 64, D = 512, I = 128, TOP = 512, GUARD = 32;
constexpr float CANARY = 12345.0f;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void sync() { ck(cudaDeviceSynchronize()); ck(cudaGetLastError()); }
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc((void**) &p, std::max<size_t>(n, 1) * sizeof(T))); }
    explicit Device(const std::vector<T>& x) : Device(x.size()) { put(x); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& x) { require(x.size() == n, "buffer shape"); if (n) ck(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> x(n); if (n) ck(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
uint32_t rng = 0x714e39u;
float random_value() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return ((int) (rng % 33) - 16) / 32.0f; }
std::vector<float> floats(size_t n) { std::vector<float> x(n); for (float& v : x) v = random_value(); return x; }
uint16_t half(float f) { return __half_as_ushort(__float2half_rn(f)); }
float unhalf(uint16_t h) { return __half2float(__ushort_as_half(h)); }
std::vector<uint16_t> halves(size_t n) { std::vector<uint16_t> x(n); for (auto& v : x) v = half(random_value()); return x; }

void check_index_row(const std::vector<float>& q, const std::vector<uint16_t>& keys,
                     const std::vector<float>& w, const std::vector<float>& scores,
                     const std::vector<int>& sel, int t, int visible, int ld) {
    std::vector<double> reference(visible);
    double max_error = 0;
    for (int j = 0; j < visible; ++j) {
        double score = 0;
        for (int h = 0; h < H; ++h) {
            double dot = 0;
            for (int k = 0; k < I; ++k) dot += (double) q[((size_t) t * H + h) * I + k] * unhalf(keys[(size_t) j * I + k]);
            score += std::max(dot, 0.0) * w[t * H + h];
        }
        reference[j] = score;
        const float got = scores[(size_t) t * ld + j];
        require(std::isfinite(got), "nonfinite index score");
        max_error = std::max(max_error, std::abs(got - score));
        require(std::abs(got - score) <= 1e-5 * (1 + std::abs(score)), "index FP64 score parity");
    }
    std::vector<int> order(visible);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const float x = scores[(size_t) t * ld + a], y = scores[(size_t) t * ld + b];
        return x > y || (x == y && a < b);
    });
    const int top = std::min(TOP, visible);
    for (int j = 0; j < top; ++j) require(sel[t * TOP + j] == order[j], "top-k ordering/tie/visibility contract");
    for (int j = top; j < TOP; ++j) require(sel[t * TOP + j] == -12345, "selection wrote beyond visible top-k");
    // Independently check membership against FP64, with a measured error envelope at near ties.
    if (top) {
        auto sorted = reference;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        const double cut = sorted[top - 1];
        for (int j = 0; j < top; ++j)
            require(reference[sel[t * TOP + j]] + 2 * max_error + 1e-12 >= cut, "top-k disagreement away from tie");
    }
}

void index_case(int p0, int T, bool batch, bool ties) {
    const int visible = (p0 + T) / 4, ld = std::max(visible, 1) + 7;
    auto q = floats((size_t) T * H * I), w = floats((size_t) T * H);
    auto keys = halves((size_t) std::max(visible, 1) * I);
    if (ties) std::fill(q.begin(), q.end(), 0.0f);
    Device<float> dq(q), dw(w), scores(std::vector<float>((size_t) T * ld + GUARD, CANARY));
    Device<uint16_t> dk(keys);
    Device<int> sel(std::vector<int>((size_t) T * TOP + GUARD, -12345));
    if (batch) ds::indexer_batch(dq.p, dk.p, dw.p, H, 4, p0, T, TOP, scores.p, ld, sel.p, nullptr);
    else ds::indexer(dq.p, dk.p, dw.p, H, I, visible, TOP, scores.p, sel.p, nullptr);
    sync();
    const auto sc = scores.get(); const auto chosen = sel.get();
    for (int t = 0; t < T; ++t) check_index_row(q, keys, w, sc, chosen, t, (p0 + t + 1) / 4, ld);
    for (size_t j = (size_t) T * ld; j < sc.size(); ++j) require(sc[j] == CANARY, "score guard overwritten");
    for (size_t j = (size_t) T * TOP; j < chosen.size(); ++j) require(chosen[j] == -12345, "selection guard overwritten");
    std::printf("{\"case\":\"indexer\",\"batch\":%s,\"p0\":%d,\"T\":%d,\"visible_max\":%d,\"ties\":%s,\"pass\":true}\n",
                batch ? "true" : "false", p0, T, visible, ties ? "true" : "false");
}

// Scalar host RoPE: adjacent pairs on the final 64 latent coordinates.
// Match the model's stored FP32 parameters; accumulate QK/softmax/PV in FP64 below.
void inverse_rope(std::vector<double>& v, const ds::Rope& r, int p) {
    constexpr int NR = 64;
    for (int i = 0; i < NR; i += 2) {
        const float extrap = (float) p * std::pow(r.theta_scale, i / 2.0f);
        const float interp = r.freq_scale * extrap;
        float theta = interp, mag = r.attn_factor;
        if (r.ext_factor != 0) {
            const float y = ((float) (i / 2) - r.corr_lo) / std::max(0.001f, r.corr_hi - r.corr_lo);
            const float mix = (1 - std::clamp(y, 0.0f, 1.0f)) * r.ext_factor;
            theta = interp * (1 - mix) + extrap * mix;
            mag *= 1 + 0.1f * std::log(1.0f / r.freq_scale);
        }
        const double c = std::cos((double) theta) * mag, s = -std::sin((double) theta) * mag;
        const int k = D - NR + i;
        const double a = v[k], b = v[k + 1];
        v[k] = a * c - b * s; v[k + 1] = a * s + b * c;
    }
}

void attention_case(int ratio, int p0, int T, bool yarn) {
    const int hist = std::min(p0, 127), ncomp = ratio ? (p0 + T) / ratio : 0;
    auto q = floats((size_t) T * H * D), sinks = floats(H);
    // Different sink regimes ensure the sink is in the denominator with zero value contribution.
    sinks[0] = 20; sinks[1] = -20;
    auto keys = halves((size_t) (hist + T) * D), comp = halves((size_t) std::max(1, ncomp) * D);
    std::vector<int> selected((size_t) T * TOP, -12345);
    for (int t = 0; t < T && ratio == 4; ++t) {
        const int vis = (p0 + t + 1) / ratio;
        // Descending IDs exercise direct noncontiguous gathers and the 512 visibility cutoff.
        for (int j = 0; j < std::min(TOP, vis); ++j) selected[t * TOP + j] = vis - 1 - j;
    }
    const auto rope = ds::make_rope(64, yarn ? 160000.0f : 10000.0f, yarn ? 1.0f / 16 : 1.0f,
                                  yarn ? 1.0f : 0.0f, 4096, 32, 1);
    Device<float> dq(q), dsink(sinks), output(std::vector<float>(q.size() + GUARD, CANARY));
    Device<uint16_t> dk(keys), dc(comp); Device<int> dsel(selected);
    ds::attend_batch(dq.p, dk.p, hist, 128, dc.p, dsel.p, TOP, ratio, dsink.p, H, D, 64, rope, p0, T, output.p, nullptr);
    sync();
    const auto actual = output.get();
    double sum_error = 0, sum_ref = 0, max_error = 0;
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t, raw = std::min(p + 1, 128), raw0 = p - raw + 1 - (p0 - hist);
        const int nc = ratio == 4 ? std::min(TOP, (p + 1) / 4) : ratio == 128 ? (p + 1) / 128 : 0;
        std::vector<const uint16_t*> rows;
        for (int j = 0; j < raw; ++j) rows.push_back(keys.data() + (size_t) (raw0 + j) * D);
        for (int j = 0; j < nc; ++j) rows.push_back(comp.data() + (size_t) (ratio == 4 ? selected[t * TOP + j] : j) * D);
        for (int h = 0; h < H; ++h) {
            std::vector<double> logits(rows.size());
            double mx = sinks[h];
            for (size_t j = 0; j < rows.size(); ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double) q[((size_t) t * H + h) * D + d] * unhalf(rows[j][d]);
                logits[j] = dot / std::sqrt((double) D); mx = std::max(mx, logits[j]);
            }
            double den = std::exp(sinks[h] - mx);
            std::vector<double> out(D, 0);
            for (size_t j = 0; j < rows.size(); ++j) {
                const double prob = std::exp(logits[j] - mx); den += prob;
                for (int d = 0; d < D; ++d) out[d] += prob * unhalf(rows[j][d]);
            }
            for (double& v : out) v /= den;
            inverse_rope(out, rope, p);
            for (int d = 0; d < D; ++d) {
                const float got = actual[((size_t) t * H + h) * D + d];
                require(std::isfinite(got), "nonfinite latent attention");
                const double e = got - out[d];
                max_error = std::max(max_error, std::abs(e)); sum_error += e * e; sum_ref += out[d] * out[d];
                require(std::abs(e) < 2e-5, "attention FP64 absolute parity");
            }
        }
    }
    const double rel = std::sqrt(sum_error / std::max(sum_ref, 1e-30));
    require(rel < 1e-4, "attention FP64 relative L2 parity");
    for (size_t j = q.size(); j < actual.size(); ++j) require(actual[j] == CANARY, "attention output guard overwritten");
    std::printf("{\"case\":\"latent_attention\",\"ratio\":%d,\"p0\":%d,\"T\":%d,\"yarn\":%s,\"relative_L2\":%.9g,\"max_abs\":%.9g,\"pass\":true}\n",
                ratio, p0, T, yarn ? "true" : "false", rel, max_error);
}

void ring_case(int p0, int T) {
    constexpr int R = 128;
    const int hist = std::min(p0, R - 1);
    auto ring = halves((size_t) R * D), keys = halves((size_t) (hist + T) * D);
    auto expected = keys;
    for (int r = 0; r < hist; ++r) std::copy_n(ring.data() + (size_t) ((p0 - hist + r) % R) * D, D, expected.data() + (size_t) r * D);
    Device<uint16_t> dr(ring), dk(keys);
    ds::win_load(dr.p, R, dk.p, hist, p0, D, nullptr); sync();
    require(dk.get() == expected, "window load wrap/chunk parity");
    for (int r = std::max(0, hist + T - R); r < hist + T; ++r)
        std::copy_n(expected.data() + (size_t) r * D, D, ring.data() + (size_t) ((p0 - hist + r) % R) * D);
    ds::win_store(dr.p, R, dk.p, hist, p0, T, D, nullptr); sync();
    require(dr.get() == ring, "window store wrap/chunk parity");
    std::printf("{\"case\":\"raw_ring\",\"p0\":%d,\"T\":%d,\"pass\":true}\n", p0, T);
}
} // namespace

int main() {
    try {
        cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, 0));
        require(p.major == 7 && p.minor == 0, "P0 V100 parity requires actual cc 7.0");
        // The workflow runs separate processes for default dispatch and legacy forced FP32.
        for (int n : {0, 1, 512, 513, 8192}) index_case(n ? n * 4 - 1 : 0, 1, false, false);
        index_case(513 * 4 - 1, 1, false, true);
        const std::pair<int, int> index_shapes[] = {{0, 5}, {2044, 9}, {4091, 33}};
        for (auto s : index_shapes) index_case(s.first, s.second, true, false);
        index_case(2044, 9, true, true);
        const std::pair<int, int> ring_shapes[] = {{0, 3}, {127, 3}, {128, 33}, {2047, 5}};
        for (auto s : ring_shapes) ring_case(s.first, s.second);
        for (bool yarn : {false, true}) {
            attention_case(0, 0, 3, yarn); attention_case(0, 127, 3, yarn);
            attention_case(4, 3, 3, yarn); attention_case(4, 2047, 5, yarn);
            attention_case(128, 127, 3, yarn); attention_case(128, 255, 3, yarn);
        }
        std::puts("{\"suite\":\"dsv4_fp32_components\",\"pass\":true}");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DeepSeek fallback parity FAILED: %s\n", e.what());
        return 1;
    }
}
