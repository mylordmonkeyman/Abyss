// Experimental P1e scorer shared by the component test and prompt integration.
// No ownership, allocation, handle mutation or synchronization: all scratch belongs
// to the caller, is exclusive until stream completion, and must not alias inputs.
#pragma once
#ifndef STRATA_USE_HIP
#include "strata/kernels/dsv4_fast.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <stdexcept>
namespace strata::kernels::dsv4::p1 {
inline void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
inline void blas(cublasStatus_t e) { if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("P1 indexer cuBLAS call failed"); }
constexpr int TT=256, KK=512, HEADS=64, DIM=128;
constexpr int ROWS_PER_WARP=1;
__global__ void unpack(const uint16_t* keys,float* out,int rows){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*DIM)out[i]=__half2float(__ushort_as_half(keys[i]));
}
// One warp per (token,key); contiguous head reads from the FP32 GEMM result.
__global__ void epilogue(const float* dots,const float* weights,float* scores,
                         int p0,int t0,int tokens,int j0,int rows,int ld){
    const int lane=threadIdx.x&31, warp=threadIdx.x/32;
    // One score per warp; the measured eight-score scheduling variant did not win.
    for(int step=0;step<ROWS_PER_WARP;++step){
        const int item=blockIdx.x*8*ROWS_PER_WARP+warp+step*8;
        if(item>=tokens*rows)break;
        const int t=item/rows,j=item%rows;
        if(j0+j>=(p0+t0+t+1)/4)continue; // Uniform warp; later iterations can see a new token.
        float sum=0;
        for(int h=lane;h<HEADS;h+=32)
            sum+=fmaxf(dots[(size_t)j*tokens*HEADS+t*HEADS+h],0.0f)*weights[(t0+t)*HEADS+h];
        for(int d=16;d;d/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(!lane)scores[(size_t)(t0+t)*ld+j0+j]=sum;
    }
}
// At most 32 MiB dot + 256 KiB key scratch; cuBLAS uses the caller's
// separately owned 16 MiB workspace. Query FP32 and key FP16 values are unchanged.
inline void run(cublasHandle_t handle,cudaStream_t stream,const float* q,
                const uint16_t* k,const float* w,int p0,int T,int ld,
                float* score,int* ids,float* key,size_t key_count,
                float* dot,size_t dot_count) {
    if(T<=0 || p0<0 || key_count<size_t(KK)*DIM ||
       dot_count<size_t(std::min(T,TT))*HEADS*KK || ld<(p0+T)/4)
        throw std::runtime_error("P1 indexer scratch/shape contract violated");
        const float one=1,zero=0;
        // Key-major loop converts each key tile once, reusing it across query tiles.
        for(int j0=0;j0<(p0+T)/4;j0+=KK){
            const int rows=std::min(KK,(p0+T)/4-j0);
            unpack<<<(rows*DIM+255)/256,256,0,stream>>>(k+(size_t)j0*DIM,key,rows);ck(cudaGetLastError());
            for(int t0=0;t0<T;t0+=TT){
                const int tokens=std::min(TT,T-t0);
                if(j0>=(p0+t0+tokens)/4)continue;
                // Query columns are contiguous 128-D FP32 vectors. C is [keys][tokens*heads].
                blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,tokens*HEADS,rows,DIM,
                    &one,q+(size_t)t0*HEADS*DIM,DIM,key,DIM,&zero,dot,tokens*HEADS));
                epilogue<<<(tokens*rows+8*ROWS_PER_WARP-1)/(8*ROWS_PER_WARP),256,0,stream>>>(dot,w,score,p0,t0,tokens,j0,rows,ld);ck(cudaGetLastError());
            }
        }
        // Reuse upstream selection literally. Separate launches are a prototype penalty;
        // timings include them and cannot be mislabeled as score-only measurements.
        for(int t=0;t<T;++t)indexer_select(score+(size_t)t*ld,(p0+t+1)/4,512,ids+(size_t)t*512,stream);
}
}
#endif
