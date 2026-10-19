// P3.12 isolated vendor-API contract. No engine dispatch is changed by this test.
// Calls the pinned ggml implementation on both sides; it checks the inverse-map
// and packed-byte contract, not an independent scalar quantization oracle.
#include "quantize.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mmq scatter: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
template<class T> T* alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T)));
    return p;
}

void run(int nt, int cols, int topk, int layer, ggml_type type, bool edge, bool timing) {
    const int rows = nt * topk, ld = cols + 16, padded = (cols + 511) / 512 * 512;
    const size_t payload = (size_t) rows * (padded / QK8_1_MMQ) * sizeof(block_q8_1_mmq);
    // Match q8_bytes's extra MMQ tail as well as external allocation guards.
    constexpr size_t guard = 256, tail = 128 * sizeof(block_q8_1_mmq);
    const size_t bytes = guard + payload + tail + guard;
    std::mt19937 rng(1234 + layer);
    std::vector<float> x((size_t) nt * ld, 9876.0f);
    for (int t = 0; t < nt; ++t) {
        for (int c = 0; c < cols; ++c) {
            float v = (int(rng() % 20001) - 10000) * 0.001f;
            if (t % 5 == 0) v = 0.0f;
            if (c % 37 == 0) v = -0.0f;
            if (c % 53 == 0) v = std::ldexp(float(int(rng() % 3) - 1), -130);
            if (c % 97 == 0) v = 65504.0f;
            if (edge && c == 17) v = std::numeric_limits<float>::quiet_NaN();
            if (edge && c == 23) v = std::numeric_limits<float>::infinity();
            if (!edge && t % 5 == 0) v = 0.0f;
            x[(size_t) t * ld + c] = v;
        }
    }
    // Sort token-slot routes by (synthetic) expert ID, exactly the domain of
    // Abyss row_tok. Multiple slots may select the same expert, but destinations
    // are unique. Duplicate destination writes are invalid/racy, not supported.
    std::vector<int32_t> order(rows), inverse(rows), gather(rows);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> expert(rows);
    for (auto& e : expert) e = int(rng() % 256);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return expert[a] < expert[b]; });
    for (int row = 0; row < rows; ++row) {
        inverse[order[row]] = row;
        gather[row] = order[row] / topk;
    }
    float* dx = alloc<float>(x.size());
    int32_t *di = alloc<int32_t>(rows), *dg = alloc<int32_t>(rows);
    auto *a = alloc<unsigned char>(bytes), *b = alloc<unsigned char>(bytes);
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(di, inverse.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dg, gather.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice));
    ck(cudaMemset(a, 0xa5, bytes)); ck(cudaMemset(b, 0xa5, bytes));
    const auto indexed = [&] {
        quantize_mmq_q8_1_cuda(dx, dg, a + guard, type, cols, ld, rows * ld, rows * ld,
                               padded, rows, 1, 1, nullptr);
        ck(cudaGetLastError());
    };
    const auto scatter = [&] {
        quantize_scatter_mmq_q8_1_cuda(dx, di, b + guard, type, cols, ld, padded,
                                       nt, rows, topk, nullptr);
        ck(cudaGetLastError());
    };
    indexed(); scatter(); ck(cudaDeviceSynchronize());
    std::vector<unsigned char> ha(bytes), hb(bytes);
    ck(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
    const auto mismatch = std::mismatch(ha.begin(), ha.end(), hb.begin());
    bool guards = true;
    for (size_t i = 0; i < bytes; ++i)
        if (i < guard || i >= guard + payload) guards &= ha[i] == 0xa5 && hb[i] == 0xa5;
    if (mismatch.first != ha.end() || !guards) {
        std::fprintf(stderr, "byte/guard failure: T=%d K=%d type=%s layer=%d offset=%zu\n",
                     nt, cols, ggml_type_name(type), layer, size_t(mismatch.first-ha.begin()));
        std::exit(1);
    }
    float indexed_ms = 0, scatter_ms = 0;
    if (timing) {
        // Five paired samples after warmup, alternating order. CUDA-event times
        // exclude maps/allocations/copies and are NOT model/TTFT improvements.
        cudaEvent_t start, stop;
        ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
        const auto measure = [&](bool sc) {
            ck(cudaEventRecord(start));
            for (int i = 0; i < 20; ++i) { if (sc) scatter(); else indexed(); }
            ck(cudaEventRecord(stop)); ck(cudaEventSynchronize(stop));
            float ms; ck(cudaEventElapsedTime(&ms, start, stop));
            return ms / 20;
        };
        std::vector<float> at, bt;
        for (int rep = 0; rep < 5; ++rep) {
            float am, bm;
            if (rep % 2) { bm=measure(true); am=measure(false); }
            else { am=measure(false); bm=measure(true); }
            at.push_back(am); bt.push_back(bm);
            std::printf("{\"kind\":\"timing_sample\",\"tokens\":%d,\"cols\":%d,\"type\":\"%s\","
                        "\"repetition\":%d,\"indexed_ms\":%.9g,\"scatter_ms\":%.9g}\n",
                        nt, cols, ggml_type_name(type), rep, am, bm);
        }
        std::sort(at.begin(),at.end()); std::sort(bt.begin(),bt.end());
        indexed_ms=at[2]; scatter_ms=bt[2];
        ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    }
    std::printf("{\"kind\":\"parity\",\"tokens\":%d,\"cols\":%d,\"topk\":%d,\"layer_seed\":%d,"
                "\"type\":\"%s\",\"layout\":%d,\"nonfinite_probe\":%s,\"byte_parity\":true,"
                "\"guards\":true,\"q8_bytes_written_each\":%zu,\"indexed_row_quantizations\":%d,"
                "\"scatter_row_quantizations\":%d,\"indexed_ms\":%.9g,\"scatter_ms\":%.9g}\n",
                nt, cols, topk, layer, ggml_type_name(type), int(mmq_get_q8_1_ds_layout(type)),
                edge ? "true":"false", payload, rows, nt, indexed_ms, scatter_ms);
    ck(cudaFree(dx)); ck(cudaFree(di)); ck(cudaFree(dg)); ck(cudaFree(a)); ck(cudaFree(b));
}
}

int main(int argc, char** argv) {
    int dev=0; ck(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,dev));
    if (prop.major != 7 || prop.minor != 0) {
        std::fprintf(stderr,"isolated V100 contract requires cc 7.0\n"); return 1;
    }
    const bool timing = argc == 2 && std::strcmp(argv[1],"--bench") == 0;
    const int widths[] = {1,2,3,8,31,64,512,2048};
    int cases=0;
    for (int nt : widths) {
        // All actual gate/up formats and MXFP4's SM70 Q8_1 route, plus the
        // DS4 and D2S6 layouts absent from the actual expert inventory.
        for (auto type : {GGML_TYPE_IQ2_XS,GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q2_K,GGML_TYPE_Q4_K}) {
            run(nt,4096,6,7,type,false,timing); ++cases;
        }
    }
    // 43 routing seeds, synthetic IDs (not captured real model routing).
    for (int layer=0;layer<43;++layer) {
        run(31,768,6,layer,layer==26||layer==42 ? GGML_TYPE_MXFP4 : GGML_TYPE_IQ2_XS,false,false); ++cases;
    }
    for (auto type : {GGML_TYPE_IQ2_XS,GGML_TYPE_Q2_K,GGML_TYPE_Q4_K}) {
        run(3,256,1,11,type,false,false); ++cases;
        run(8,768,6,11,type,true,false); ++cases;
    }
    std::printf("{\"kind\":\"summary\",\"cases\":%d,\"passed\":true,\"production_enabled\":false}\n",cases);
}
