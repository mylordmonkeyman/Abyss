// src/kernels/cuda/dsa_topk.cuh - the DSA indexer's choice of the top pools in O(n_vis) (one block, 1024 threads).
//
// The order it writes is the one the O(n_vis^2) rank loop wrote (score descending, ties by the lower pool index), so
// the selected cells - and every result built on them - are unchanged; only the cost stops growing with the square of
// the context (at 30k tokens the old loop was ~56M comparisons per DSA layer per token, in one block).
//   1. a radix select (4 passes of 8 bits) over the scores' order-preserving uint32 keys finds the cut-off key - the
//      top-th largest - and how many of the keys equal to it are taken;
//   2. one pass in pool order (each thread a contiguous run, exclusive scans for the offsets) collects the winners:
//      every key above the cut-off and the first `need` equal to it - the lower indices, as the rank loop chose;
//   3. a bitonic sort of the <= 1024 winners, packed as (key << 32) | ~index, descending.
// CUDA builds (v1.3.3) take the same cut-off and write the same cells, faster (one block at 2k keys: 18 -> 12 us; at 31k:
// 35 -> 25, RTX 5090): three radix passes of 12, 12 and 8 bits, the cut-off bin found by a block scan instead of a
// serial walk by thread 0; the winners above the cut collected in any order with coalesced reads (the ones equal to it
// too when all of them are taken, else in pool order as before); and each winner's rank counted (how many winners are
// larger - the values are distinct) instead of the bitonic sort. HIP keeps the version above (it assumes no warp width).
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels::dsa {

// a float's order-preserving key: larger float, larger key (-0 is +0; the scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis) in rank order; 1 <= top <= 1024,
// top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that includes
// this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[NT], s_eq[NT];
    __shared__ unsigned long long s_win[NT];
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
    }
    __syncthreads();
    // 1. the cut-off key, a byte at a time from the top
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_vis; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need;
            for (int b = 255; b >= 0; --b) {
                const int c = (int) s_hist[b];
                if (c >= need) {
                    s_prefix = prefix | ((unsigned) b << shift);
                    s_mask = mask | (255u << shift);
                    s_need = need;
                    break;
                }
                need -= c;
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    // 2. the winners, in pool order
    const int chunk = (n_vis + NT - 1) / NT;
    const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    s_gt[tid] = gt;
    s_eq[tid] = eq;
    __syncthreads();
    for (int off = 1; off < NT; off <<= 1) {   // inclusive scans of both counts
        int a = 0, b = 0;
        if (tid >= off) {
            a = s_gt[tid - off];
            b = s_eq[tid - off];
        }
        __syncthreads();
        s_gt[tid] += a;
        s_eq[tid] += b;
        __syncthreads();
    }
    const int n_gt = s_gt[NT - 1];   // == top - need_eq
    int og = s_gt[tid] - gt, oe = s_eq[tid] - eq;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) {
            s_win[og++] = w;
        } else if (k == cut) {
            if (oe < need_eq) s_win[n_gt + oe] = w;
            ++oe;
        }
    }
    // 3. descending bitonic sort, padded with the smallest value
    int n2 = 1;
    while (n2 < top) n2 <<= 1;
    for (int i = top + tid; i < n2; i += NT) s_win[i] = 0ull;
    __syncthreads();
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n2; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const unsigned long long a = s_win[i], b = s_win[j];
                    if ((i & size) == 0 ? a < b : a > b) {
                        s_win[i] = b;
                        s_win[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int r = tid; r < top; r += NT) {
        const int p = (int) ~(unsigned) (s_win[r] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
}

#else

// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis) in rank order; 1 <= top <= 1024,
// top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that includes
// this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    __shared__ unsigned s_hist[4096];
    __shared__ unsigned s_wsum[32];
    __shared__ unsigned s_prefix, s_mask, s_eq_all;
    __shared__ int s_need, s_ngt, s_neq;
    __shared__ int s_eq[NT];
    __shared__ unsigned long long s_win[NT];
    __shared__ unsigned short s_rank[2][NT];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
        s_ngt = 0;
        s_neq = 0;
    }
    __syncthreads();
    // 1. the cut-off key in three passes of 12, 12 and 8 bits from the top
    for (int pass = 0; pass < 3; ++pass) {
        const int bits = pass < 2 ? 12 : 8, shift = pass == 0 ? 20 : pass == 1 ? 8 : 0;
        const unsigned nb = 1u << bits, bmask = nb - 1u;
        for (int i = tid; i < (int) nb; i += NT) s_hist[i] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_vis; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & bmask], 1u);
        }
        __syncthreads();
        // the bin that holds the need-th largest: thread t sums the 4 bins nb-1-4t .. nb-4-4t (from the top), then a
        // block scan of those sums
        const int per = (int) nb / NT;   // 4, or 0 for the 8-bit pass (256 bins: one a thread for t < 256)
        unsigned c4[4] = {0u, 0u, 0u, 0u}, own = 0u;
        if (per == 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                c4[j] = s_hist[nb - 1u - (unsigned) (4 * tid + j)];
                own += c4[j];
            }
        } else if (tid < (int) nb) {
            c4[0] = s_hist[nb - 1u - (unsigned) tid];
            own = c4[0];
        }
        unsigned incl = own;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 31) s_wsum[wid] = incl;
        __syncthreads();
        if (wid == 0) {
            unsigned v = s_wsum[lane], x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            s_wsum[lane] = x - v;   // exclusive: the warps before
        }
        __syncthreads();
        incl += s_wsum[wid];
        const unsigned need = (unsigned) s_need;
        unsigned before = incl - own;
        if (before < need && incl >= need) {
            const int m = per == 4 ? 4 : 1;
            for (int j = 0; j < m; ++j) {
                if (before + c4[j] >= need) {
                    const unsigned bin = nb - 1u - (unsigned) (per == 4 ? 4 * tid + j : tid);
                    s_prefix = prefix | (bin << shift);
                    s_mask = mask | (bmask << shift);
                    s_need = (int) (need - before);
                    s_eq_all = c4[j];   // after the last pass: how many keys equal the cut
                    break;
                }
                before += c4[j];
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    const int n_gt = top - need_eq;
    // 2. the winners. Every key above the cut, in any order (the sort below orders them); the keys equal to it in any
    // order too when all of them are taken, else the first need_eq in pool order (a contiguous run a thread, scans)
    const bool all_eq = (int) s_eq_all == need_eq;
    for (int i = tid; i < n_vis; i += NT) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) s_win[atomicAdd(&s_ngt, 1)] = w;
        else if (k == cut && all_eq) s_win[n_gt + atomicAdd(&s_neq, 1)] = w;
    }
    if (!all_eq) {
        const int chunk = (n_vis + NT - 1) / NT;
        const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
        int eq = 0;
        for (int i = lo; i < hi; ++i) eq += score_key(sc[i]) == cut;
        s_eq[tid] = eq;
        __syncthreads();
        for (int off = 1; off < NT; off <<= 1) {
            int b = 0;
            if (tid >= off) b = s_eq[tid - off];
            __syncthreads();
            s_eq[tid] += b;
            __syncthreads();
        }
        int oe = s_eq[tid] - eq;
        for (int i = lo; i < hi && oe < need_eq; ++i) {
            const unsigned k = score_key(sc[i]);
            if (k == cut) {
                s_win[n_gt + oe] = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
                ++oe;
            }
        }
    }
    __syncthreads();
    // 3. the rank of each winner: how many are larger (the values are distinct: the index is in the low bits)
    {
        const int i = tid % top, half = tid / top;   // top <= 512: two threads a winner, each half of the others
        if (top <= NT / 2) {
            if (tid < 2 * top) {
                const unsigned long long v = s_win[i];
                const int j0 = half ? top / 2 : 0, j1 = half ? top : top / 2;
                int r = 0;
                for (int j = j0; j < j1; ++j) r += s_win[j] > v;
                s_rank[half][i] = (unsigned short) r;
            }
        } else if (tid < top) {
            const unsigned long long v = s_win[tid];
            int r = 0;
            for (int j = 0; j < top; ++j) r += s_win[j] > v;
            s_rank[0][tid] = (unsigned short) r;
            s_rank[1][tid] = 0;
        }
    }
    __syncthreads();
    if (tid < top) {
        const int r = s_rank[0][tid] + s_rank[1][tid];
        const int p = (int) ~(unsigned) (s_win[tid] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
}

#endif

}  // namespace strata::kernels::dsa
