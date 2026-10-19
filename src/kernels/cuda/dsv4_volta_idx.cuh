#pragma once
// P1 candidate, used only by the isolated scorer executable until model gates pass.
#include "volta_mma884.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels::volta {
// Four independent quad pairs score 32 keys against eight heads per warp.
// There is no global [keys, heads] temporary. All lanes participate in MMA,
// including masked key tails. Queries retain independent hi/lo accumulators.
template<int Warps>
__global__ void __launch_bounds__(Warps * 32) idx_score_volta_kernel(
    const float* iq, const uint16_t* keys, const float* weights, int visible, float* score) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    static_assert(Warps == 4 || Warps == 8, "tested CTA sizes only");
    // Padding breaks the 128-half row stride's same-bank head reads. Each
    // quad pair reads eight different heads at the same K coordinate.
    constexpr int QStride = 130;
    __shared__ uint16_t qhi[64 * QStride], qlo[64 * QStride];
    __shared__ float w[64];
    for (int i = threadIdx.x; i < 64 * 128; i += Warps * 32) {
        const __half hi = __float2half_rn(iq[i]);
        const int dst = (i / 128) * QStride + i % 128;
        qhi[dst] = __half_as_ushort(hi);
        qlo[dst] = __half_as_ushort(__float2half_rn(iq[i] - __half2float(hi)));
    }
    for (int i = threadIdx.x; i < 64; i += Warps * 32) w[i] = weights[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3, j = (lane & 3) + 4 * (lane >> 4);
    const int row0 = (blockIdx.x * Warps + warp) * 32 + qp * 8;
    float total[2] = {};
    for (int head0 = 0; head0 < 64; head0 += 8) {
        float hi[8] = {}, lo[8] = {};
        for (int k0 = 0; k0 < 128; k0 += 4) {
            uint32_t a[2] = {}, bh[2] = {}, bl[2] = {};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int shift = 16 * (k & 1), pair = k >> 1;
                const uint16_t key = row0 + j < visible ? keys[(size_t)(row0 + j) * 128 + k0 + k] : 0;
                a[pair] |= uint32_t(key) << shift;
                bh[pair] |= uint32_t(qhi[(head0 + j) * QStride + k0 + k]) << shift;
                bl[pair] |= uint32_t(qlo[(head0 + j) * QStride + k0 + k]) << shift;
            }
            mma884(hi, a[0], a[1], bh[0], bh[1]);
            mma884(lo, a[0], a[1], bl[0], bl[1]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float sum = 0;
#pragma unroll
            for (int band = 0; band < 2; ++band) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int i = 2 * r + 4 * band + c;
                    // ReLU before each learned head weight, never after the head sum.
                    sum += fmaxf(hi[i] + lo[i], 0.0f) * w[head0 + (lane & 2) + 4 * band + c];
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            total[r] += sum;
        }
    }
    if (!(lane & 2)) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + (lane & 1) + 4 * (lane >> 4) + 2 * r;
            if (row < visible) score[row] = total[r];
        }
    }
#else
    // The isolated host launcher rejects devices other than SM70.
    (void) iq; (void) keys; (void) weights; (void) visible; (void) score;
#endif
}
} // namespace strata::kernels::volta
