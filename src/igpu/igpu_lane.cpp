// src/igpu/igpu_lane.cpp - the iGPU lane (include/strata/igpu/igpu_lane.h): glm_fast's HIP expert kernels on the
// APU's integrated GPU, reading the engine's RAM-tier slots in place.  One call: x (host) -> q8_1 -> gate/up ->
// down, the k experts' weighted sum -> out (host).  Built by hipcc with the HIP compat header (cuda* -> hip*), as one
// translation unit with glm_fast.cu (its row-dot helpers are file-local).
#include "strata/igpu/igpu_lane.h"
#include "../kernels/cuda/glm_fast.cu"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace strata::kernels::glmf {
namespace {
// The lane's down for a FEW experts (a decode's lane call holds 1-2): every warp of a block on NR rows of its own,
// looping over the experts and combining in plan order as moe_down does (the same sum).  moe_down gives each expert
// one warp of a block's 8 (the device's plans hold 8), so at k = 1 seven of them idle: ~55 us for one expert's 3 MB.
template<int TD, int NR>
__global__ void __launch_bounds__(256) lane_down_kernel(const unsigned long long* plan_ptr, const float* plan_w, int k,
                                                        int n_ff, size_t down_off, const block_q8_1* __restrict__ hq,
                                                        float* __restrict__ out) {
    __shared__ unsigned long long s_ptr[8];
    __shared__ float s_w[8];
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    if (threadIdx.x < 8) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? plan_ptr[threadIdx.x] : 0ull;
        s_w[threadIdx.x] = (int) threadIdx.x < k ? plan_w[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + warp) * NR;
    const size_t drb = rbytes<TD>(n_ff);
    float m[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) m[j] = 0.0f;
    for (int i = 0; i < k; ++i) {
        float s[NR];
        if (s_ptr[i] != 0ull) {
            const uint8_t* rows[NR];
#pragma unroll
            for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) s_ptr[i] + down_off + (size_t) (r0 + j) * drb;
            rows_dot<TD, NR>(rows, hq + (size_t) i * (n_ff / 32), n_ff, lane, s_tab, s);
        } else {
#pragma unroll
            for (int j = 0; j < NR; ++j) s[j] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < NR; ++j) m[j] += s_w[i] * s[j];
    }
    if (lane == 0)
#pragma unroll
        for (int j = 0; j < NR; ++j) out[r0 + j] = m[j] + 0.0f;
}
}  // namespace

bool lane_down(int d_type, const unsigned long long* plan_ptr, const float* plan_w, int k, int n_embd, int n_ff,
               size_t down_off, const void* hq, float* out, hipStream_t s) {
    constexpr int NR = DOWN_ROWS;
    if (n_embd % (8 * NR) != 0) return false;
    const int blocks = n_embd / (8 * NR);
    const auto* H = (const block_q8_1*) hq;
    switch (d_type) {
#define LANE_DN(T) \
        case T: lane_down_kernel<T, NR><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out); break;
        LANE_DN(16) LANE_DN(18) LANE_DN(19) LANE_DN(23) LANE_DN(10) LANE_DN(11) LANE_DN(17) LANE_DN(22) LANE_DN(21)
        LANE_DN(29) LANE_DN(12) LANE_DN(13) LANE_DN(14) LANE_DN(39)
#undef LANE_DN
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}
__global__ void lane_poke_kernel() {}
void lane_poke(hipStream_t s) { lane_poke_kernel<<<1, 32, 0, s>>>(); }
// ABYSS_IGPU_TEST_STALL (test only): spins `ticks` of the device's wall clock, standing in for a queue the kernel
// driver holds while it re-validates registered memory
__global__ void lane_stall_kernel(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
}
}  // namespace strata::kernels::glmf

namespace {
constexpr size_t kQ8 = 36;
struct Range {
    const uint8_t* host;
    size_t bytes;
    uint8_t* dev;
};
struct Plan {                         // host-mapped: the kernels read it in place
    unsigned long long ptr[8];
    float w[8];
    int cpu_flag;
};
struct Lane {
    bool open = false;
    int n_embd = 0, n_ff = 0;
    hipStream_t s = nullptr;
    float *x_h = nullptr, *out_h = nullptr, *x_d = nullptr, *out_d = nullptr;
    Plan *plan_h = nullptr, *plan_d = nullptr;
    void *xq = nullptr, *hq = nullptr;
    std::vector<Range> ranges;
    std::mutex mu;
    // abyss_igpu_rows: the experts' table (host-mapped) and the scratch, grown as needed
    unsigned long long* rptr_h = nullptr;
    int* rlight_h = nullptr;
    int r_cap = 0;
    void *rxq = nullptr, *rhq = nullptr;
    size_t rxq_b = 0, rhq_b = 0;
    // A call that has not finished by its deadline is left running and answered by the CPU lane (return 1); the calls
    // after it go to the CPU too until it has (the queue may be held for seconds - see abyss_igpu_experts)
    bool pending = false;
    std::chrono::steady_clock::time_point t_pend;
    unsigned long long on_cpu = 0;
};
Lane L;

// the device address of p inside a registered range, or 0
unsigned long long dev_of(const void* p) {
    const uint8_t* b = (const uint8_t*) p;
    auto it = std::upper_bound(L.ranges.begin(), L.ranges.end(), b, [](const uint8_t* q, const Range& r) { return q < r.host; });
    if (it != L.ranges.begin() && b < (--it)->host + it->bytes) return (unsigned long long) (it->dev + (b - it->host));
    return 0ull;
}
void say(char* err, int n, const char* what, hipError_t e) {
    if (err != nullptr && n > 0) std::snprintf(err, (size_t) n, "%s: %s", what, hipGetErrorString(e));
}
}  // namespace

extern "C" {

__attribute__((visibility("default"))) int abyss_igpu_version(void) { return ABYSS_IGPU_API; }

__attribute__((visibility("default"))) int abyss_igpu_open(int n_embd, int n_ff, char* err, int err_n) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (L.open) return 0;
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess || n < 1) {
        say(err, err_n, "no HIP device", e == hipSuccess ? hipErrorNoDevice : e);
        return -1;
    }
    hipSetDeviceFlags(hipDeviceScheduleSpin);   // the service thread waits ~0.1 ms a call: spin, not sleep
    if ((e = hipSetDevice(0)) != hipSuccess) { say(err, err_n, "hipSetDevice", e); return -1; }
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    const auto hm = [&](void** p, size_t b) { return hipHostMalloc(p, b, hipHostMallocMapped); };
    if ((e = hipStreamCreateWithFlags(&L.s, hipStreamNonBlocking)) != hipSuccess ||
        (e = hm((void**) &L.x_h, (size_t) n_embd * sizeof(float))) != hipSuccess ||
        (e = hm((void**) &L.out_h, (size_t) n_embd * sizeof(float))) != hipSuccess ||
        (e = hm((void**) &L.plan_h, sizeof(Plan))) != hipSuccess ||
        (e = hipMalloc(&L.xq, (size_t) n_embd / 32 * kQ8)) != hipSuccess ||
        (e = hipMalloc(&L.hq, (size_t) 8 * n_ff / 32 * kQ8)) != hipSuccess) {
        say(err, err_n, "iGPU lane buffers", e);
        return -1;
    }
    hipHostGetDevicePointer((void**) &L.x_d, L.x_h, 0);
    hipHostGetDevicePointer((void**) &L.out_d, L.out_h, 0);
    hipHostGetDevicePointer((void**) &L.plan_d, L.plan_h, 0);
    std::memset(L.plan_h, 0, sizeof(Plan));
    L.open = true;
    return 0;
}

__attribute__((visibility("default"))) int abyss_igpu_register(const void* host, size_t bytes, char* err, int err_n) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) { say(err, err_n, "iGPU lane not open", hipErrorNotInitialized); return -1; }
    hipError_t e = hipHostRegister(const_cast<void*>(host), bytes, hipHostRegisterMapped);
    if (e != hipSuccess) { say(err, err_n, "hipHostRegister", e); return -1; }
    void* d = nullptr;
    if ((e = hipHostGetDevicePointer(&d, const_cast<void*>(host), 0)) != hipSuccess) {
        say(err, err_n, "hipHostGetDevicePointer", e);
        return -1;
    }
    // kept sorted by host address: a call looks its experts up by binary search (the RAM tier is registered in a few
    // hundred pieces)
    const Range nr{(const uint8_t*) host, bytes, (uint8_t*) d};
    L.ranges.insert(std::upper_bound(L.ranges.begin(), L.ranges.end(), nr,
                                     [](const Range& a, const Range& b) { return a.host < b.host; }),
                    nr);
    return 0;
}

__attribute__((visibility("default"))) int abyss_igpu_experts(int gu_type, int d_type, int n_embd, int n_ff,
                                                              size_t down_off, float limit, int ne,
                                                              const void* const* blobs, const float* w,
                                                              const float* x, float* out) {
    if (!L.open || ne < 1 || ne > 8 || n_embd > L.n_embd || n_ff > L.n_ff) return 1;
    // A late call still owns the buffers and the queue: the CPU takes this one, until it is done.  The kernel driver
    // can hold the iGPU's queues for seconds while it faults registered memory in again (seen: 0.7-18 s, a few times
    // after the host had been idle for hours; kernel log "svm_range_restore_work / amdgpu_amdkfd_restore_userptr_worker
    // hogged CPU"), and a decode waited all of it for one layer's experts.
    if (L.pending) {
        if (hipStreamQuery(L.s) != hipSuccess) {
            ++L.on_cpu;
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - L.t_pend).count();
        std::fprintf(stderr, "iGPU lane: the late call finished after %.0f ms; %llu calls went to the CPU meanwhile\n", ms,
                     L.on_cpu);
        L.pending = false;
        L.on_cpu = 0;
    }
    for (int i = 0; i < ne; ++i) {
        const uint8_t* b = (const uint8_t*) blobs[i];
        unsigned long long d = 0ull;
        auto it = std::upper_bound(L.ranges.begin(), L.ranges.end(), b,
                                   [](const uint8_t* p, const Range& r) { return p < r.host; });
        if (it != L.ranges.begin() && b < (--it)->host + it->bytes) d = (unsigned long long) (it->dev + (b - it->host));
        if (d == 0ull) return 1;
        L.plan_h->ptr[i] = d;
        L.plan_h->w[i] = w[i];
    }
    for (int i = ne; i < 8; ++i) { L.plan_h->ptr[i] = 0ull; L.plan_h->w[i] = 0.0f; }
    L.plan_h->cpu_flag = 0;
    std::memcpy(L.x_h, x, (size_t) n_embd * sizeof(float));
    gf::MoeDev md;
    md.plan_ptr = L.plan_d->ptr;
    md.plan_w = L.plan_d->w;
    md.cpu_flag = &L.plan_d->cpu_flag;
    gf::quantize_q8_1(L.x_d, L.xq, n_embd, L.s);
    gf::moe_gate_up(gu_type, md, ne, n_embd, n_ff, limit, L.xq, L.hq, nullptr, 0, nullptr, 0, nullptr, L.s);
    if (!gf::lane_down(d_type, L.plan_d->ptr, L.plan_d->w, ne, n_embd, n_ff, down_off, L.hq, L.out_d, L.s))
        gf::moe_down(d_type, md, ne, n_embd, n_ff, down_off, L.hq, nullptr, L.out_d, L.s);
    static const unsigned long long stall_ticks = [] {
        const char* v = getenv("ABYSS_IGPU_TEST_STALL");   // "<ms>:<every n calls>"
        int rate = 0;
        if (v == nullptr || hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0) != hipSuccess) return 0ull;
        return (unsigned long long) (std::atof(v) * (double) rate);   // rate: kHz
    }();
    static const unsigned long long stall_every = [] {
        const char* v = getenv("ABYSS_IGPU_TEST_STALL");
        const char* c = v ? std::strchr(v, ':') : nullptr;
        return c ? std::max(1ull, std::strtoull(c + 1, nullptr, 10)) : 0ull;
    }();
    static unsigned long long n_calls = 0;
    if (stall_ticks > 0 && stall_every > 0 && ++n_calls % stall_every == 0)
        gf::lane_stall_kernel<<<1, 32, 0, L.s>>>(stall_ticks);
    // wait (spinning: a call is ~0.05-0.3 ms) up to the deadline, STRATA_GLM_IGPU_TIMEOUT_MS (20; 0: no deadline)
    static const double timeout_ms = [] {
        const char* v = getenv("STRATA_GLM_IGPU_TIMEOUT_MS");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 20.0;
    }();
    const auto t0 = std::chrono::steady_clock::now();
    hipError_t e;
    while ((e = hipStreamQuery(L.s)) == hipErrorNotReady) {
        if (timeout_ms > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
            L.pending = true;
            L.t_pend = t0;
            L.on_cpu = 0;
            std::fprintf(stderr, "iGPU lane: a call is still running after %.0f ms - the CPU lane takes the calls until "
                                 "it finishes\n", timeout_ms);
            return 1;
        }
    }
    if (e != hipSuccess) {
        std::fprintf(stderr, "iGPU lane: %s\n", hipGetErrorString(e));
        return -1;
    }
    std::memcpy(out, L.out_h, (size_t) n_embd * sizeof(float));
    return 0;
}

__attribute__((visibility("default"))) void abyss_igpu_unregister(const void* host) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) return;
    for (size_t i = 0; i < L.ranges.size(); ++i)
        if (L.ranges[i].host == (const uint8_t*) host) {
            hipStreamSynchronize(L.s);
            hipHostUnregister(const_cast<void*>(host));
            L.ranges.erase(L.ranges.begin() + (long) i);
            return;
        }
}

__attribute__((visibility("default"))) int abyss_igpu_rows(int gu_type, int d_type, int n_embd, int n_ff,
                                                           size_t down_off, float limit, int ne,
                                                           const void* const* blobs, const int* r0, const int* nr,
                                                           const int* row_tok, const float* x, int n_tok, int rlo,
                                                           int rhi, float* out, int ld) {
    if (!L.open || ne < 1 || rhi <= rlo || ld < n_embd || ld < 2 * n_ff) return 1;
    if (L.pending) {   // a late decode call still holds the queue (abyss_igpu_experts)
        if (hipStreamQuery(L.s) != hipSuccess) return 1;
        L.pending = false;
    }
    const unsigned long long xd = dev_of(x), td = dev_of(row_tok), od = dev_of(out);
    if (xd == 0ull || td == 0ull || od == 0ull || dev_of(x + (size_t) n_tok * n_embd - 1) != xd + ((size_t) n_tok * n_embd - 1) * sizeof(float) ||
        dev_of(out + (size_t) rhi * ld - 1) != od + ((size_t) rhi * ld - 1) * sizeof(float))
        return 1;
    hipError_t e = hipSuccess;
    if (ne > L.r_cap) {
        if (L.rptr_h) hipHostFree(L.rptr_h);
        if (L.rlight_h) hipHostFree(L.rlight_h);
        L.rptr_h = nullptr;
        L.rlight_h = nullptr;
        L.r_cap = 0;
        if ((e = hipHostMalloc((void**) &L.rptr_h, (size_t) ne * sizeof(unsigned long long), hipHostMallocMapped)) != hipSuccess ||
            (e = hipHostMalloc((void**) &L.rlight_h, (size_t) ne * 3 * sizeof(int), hipHostMallocMapped)) != hipSuccess)
            return -1;
        L.r_cap = ne;
    }
    const size_t xq_b = (size_t) n_tok * (n_embd / 32) * kQ8, hq_b = (size_t) (rhi - rlo) * (n_ff / 32) * kQ8;
    const auto grow = [&](void*& p, size_t& have, size_t need) {
        if (need <= have) return hipSuccess;
        if (p) hipFree(p);
        p = nullptr;
        have = 0;
        const hipError_t r = hipMalloc(&p, need + need / 4);
        if (r == hipSuccess) have = need + need / 4;
        return r;
    };
    if ((e = grow(L.rxq, L.rxq_b, xq_b)) != hipSuccess || (e = grow(L.rhq, L.rhq_b, hq_b)) != hipSuccess) return -1;
    for (int i = 0; i < ne; ++i) {
        const unsigned long long d = dev_of(blobs[i]);
        if (d == 0ull) return 1;
        L.rptr_h[i] = d;
        L.rlight_h[3 * i] = i;
        L.rlight_h[3 * i + 1] = r0[i];
        L.rlight_h[3 * i + 2] = nr[i];
    }
    unsigned long long pd = 0ull, ld_d = 0ull;
    hipHostGetDevicePointer((void**) &pd, L.rptr_h, 0);
    hipHostGetDevicePointer((void**) &ld_d, L.rlight_h, 0);
    if (!gf::rows_experts(gu_type, d_type, nullptr, 0, down_off, (const int*) ld_d, ne, (const int*) td,
                          (const float*) xd, n_tok, n_embd, n_ff, limit, rlo, rhi, L.rxq, L.rhq, (float*) od, ld, L.s,
                          (const unsigned long long*) pd))
        return 1;
    if ((e = hipStreamSynchronize(L.s)) != hipSuccess) {
        std::fprintf(stderr, "iGPU lane (prompt rows): %s\n", hipGetErrorString(e));
        return -1;
    }
    return 0;
}

__attribute__((visibility("default"))) void abyss_igpu_poke(void) {
    if (L.open && hipStreamQuery(L.s) == hipSuccess) gf::lane_poke(L.s);
}

__attribute__((visibility("default"))) void abyss_igpu_close(void) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) return;
    hipStreamSynchronize(L.s);
    for (const Range& r : L.ranges) hipHostUnregister(const_cast<uint8_t*>(r.host));
    L.ranges.clear();
    hipFree(L.xq);
    hipFree(L.hq);
    if (L.rxq) hipFree(L.rxq);
    if (L.rhq) hipFree(L.rhq);
    if (L.rptr_h) hipHostFree(L.rptr_h);
    if (L.rlight_h) hipHostFree(L.rlight_h);
    L.rxq = L.rhq = nullptr;
    L.rxq_b = L.rhq_b = 0;
    L.rptr_h = nullptr;
    L.rlight_h = nullptr;
    L.r_cap = 0;
    hipHostFree(L.x_h);
    hipHostFree(L.out_h);
    hipHostFree(L.plan_h);
    hipStreamDestroy(L.s);
    L.open = false;
}

}  // extern "C"
