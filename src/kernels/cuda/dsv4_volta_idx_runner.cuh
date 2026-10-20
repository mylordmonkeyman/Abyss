#pragma once
// Shared isolated-test / explicit experimental model entry point.
#ifndef STRATA_USE_HIP
#include "dsv4_volta_idx_batch.cuh"
#include "strata/kernels/dsv4_fast.hpp"
#include <stdexcept>
namespace strata::kernels::volta {
__global__ void prepare_scaled_queries(const float* q,int T,uint32_t* packed,int* shifts){
    const int t=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(t>=T)return;
    __shared__ float maxima[8];__shared__ int shift;
    float maximum=0;
    for(int i=threadIdx.x;i<64*128;i+=256)maximum=fmaxf(maximum,fabsf(q[(size_t)t*64*128+i]));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(threadIdx.x==0){
        float peak=0;for(int i=0;i<8;++i)peak=fmaxf(peak,maxima[i]);
        int exponent=0;if(peak>0)frexpf(peak,&exponent);shift=-exponent;shifts[t]=shift;
    }
    __syncthreads();
    const bool fast=shift>=-126&&shift<=126;
    const float factor=fast?__uint_as_float(unsigned(127+shift)<<23):0;
    for(int i=threadIdx.x;i<64*128;i+=256){
        float value=q[(size_t)t*64*128+i];
        value=fast?__fmul_rn(value,factor):scalbnf(value,shift);
        const __half hi=__float2half_rn(value),lo=__float2half_rn(value-__half2float(hi));
        packed[(size_t)t*64*128+i]=uint32_t(__half_as_ushort(hi))|(uint32_t(__half_as_ushort(lo))<<16);
    }
}
inline void indexer_batch_scaled(const float* q,const uint16_t* k,const float* w,
    int p0,int T,int ld,float* scores,int* ids,cudaStream_t stream,
    uint32_t* packed,size_t packed_count,int* shifts,size_t shift_count,bool select=true){
    if(!q||!k||!w||!scores||!ids||!packed||!shifts||p0<0||T<=0||ld<(p0+T)/4||
       packed_count<(size_t)T*64*128||shift_count<(size_t)T)
        throw std::runtime_error("scaled MMA indexer shape/pointer contract violated");
    prepare_scaled_queries<<<T,256,0,stream>>>(q,T,packed,shifts);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("scaled MMA query preparation failed");
    const dim3 grid(((p0+T)/4+127)/128,(T+7)/8);
    // A positive common token scale commutes through every per-head ReLU and
    // weighted sum in real arithmetic. Restore once per row; independently
    // test floating-point reduction/underflow effects against F32 and FP64.
    idx_score_volta_batch_kernel<8,false,true><<<grid,128,0,stream>>>(q,k,w,p0,T,ld,scores,packed,shifts);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("scaled MMA indexer launch failed");
    // Literal upstream selector, including its causal mask and tie rule.
    if(select)for(int t=0;t<T;++t)strata::kernels::dsv4::indexer_select(
        scores+(size_t)t*ld,(p0+t+1)/4,512,ids+(size_t)t*512,stream);
}
}
#endif
