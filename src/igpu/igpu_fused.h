// src/igpu/igpu_fused.h - the prompt path's host experts on the iGPU as one fused int8-WMMA pass per product
// (gate/up + clipped SwiGLU -> int8 H, then down), from Maya's src/prefill/moe_fused_rdna4.cu (mw00/project-maya,
// MIT; itself from Strata's moe_fused_iq.cu).  HIP only, built into libabyss_igpu.so beside igpu_lane.cpp.
//
// Fixed geometry: n_embd 4096, n_ff 2048 (gate and up as separate row blocks: gate rows, then up rows at up_off).
// Weight formats (ggml type ids): gate/up IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22, IQ4_XS 23;
// down IQ3_XXS 18, IQ4_NL 20, IQ2_S 22.  DeepSeek V4 Flash's experts are IQ2_XS / IQ3_XXS.
#pragma once
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace abyss::igpu_fused {

struct Geometry {
    int gu_type = -1, d_type = -1;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0;   // bytes: a gate/up row, a down row, up's and down's start
    float limit = 0.0f;                                        // the SwiGLU clamp
};

/// The device (gfx1151 so far) runs the WMMA kernels and STRATA_GLM_IGPU_FUSED is not 0.
bool available(int dev);
/// The formats and geometry the kernels cover.
bool supported(int gu_type, int d_type, int n_embd, int n_ff);
/// Bytes of the int8 activations of rows x cols (80 bytes per 64 values).
size_t act_bytes(size_t rows, int cols);
/// Bytes a gate/up or down row of the format takes for k values (0: not a covered format).
size_t row_bytes(int type, int k);
/// x (rows x cols floats, contiguous, device-readable) -> the int8 activations xa (device memory).
void quantize(const float* x, size_t rows, int cols, void* xa, hipStream_t s);
/// Fill tiles (int2 {expert, first row}) for the experts' row ranges bounds[0..ne]; returns the tile count (the
/// rows of expert e are [bounds[e], bounds[e + 1]), relative to bounds[0]; cap: the room in tiles).  < 0: cap too small.
int make_tiles(const int* bounds, int ne, int2* tiles, int cap);
/// The products: expert e's blob at blobs[e] (device addresses), rows [bounds[e], bounds[e + 1]) relative to bounds[0];
/// GU reads xa row src[r] (src indexed by the relative row), writes H row r into ha (act_bytes(rows, 2048));
/// down writes out row r (4096 floats, 16-byte aligned).  blobs, bounds and tiles are read by the kernels (device or
/// host-mapped memory).  false: not launched (a format or geometry not covered).
bool experts(const unsigned long long* blobs, int ne, const int* bounds, const int2* tiles, int ntiles,
             const Geometry& geo, const void* xa, const int* src, void* ha, float* out, hipStream_t s);

}  // namespace abyss::igpu_fused
