#pragma once
// Isolated P1 batched experiment. No production include/dispatch.
#ifndef STRATA_USE_HIP
#include "volta_mma884.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
namespace strata::kernels::volta {
// Tokens=1 is the correctness-first batched grid. Tokens=8 holds eight
// independent hi/lo accumulator chains and reuses each packed K fragment
// across those queries. Four warps/CTA, 128 compressed rows/CTA. Query shared
// memory holds only the current eight-head band, padded to 130 half elements.
// No global [token,key,head] intermediate and no new arithmetic in the engine.
template<int Tokens,bool Scaled=false,bool Packed=false>
__global__ void __launch_bounds__(128) idx_score_volta_batch_kernel(
    const float* iq,const uint16_t* keys,const float* weights,
    int p0,int T,int ld,float* score,const uint32_t* packed=nullptr,const int* packed_shifts=nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    static_assert(Tokens==1 || Tokens==8,"bounded experimental query tiles");
    constexpr int QStride=130;
    __shared__ uint16_t qhi[Tokens*8*QStride],qlo[Tokens*8*QStride];
    __shared__ float w[Tokens*8];
    __shared__ int shifts[Scaled?Tokens*8:1];
    __shared__ float scales[Scaled?Tokens*8:1],inverse_scales[Scaled?Tokens*8:1];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int qp=(lane>>2)&3,j=(lane&3)+4*(lane>>4);
    const int row0=(blockIdx.x*4+warp)*32+qp*8;
    const int t0=blockIdx.y*Tokens,max_vis=(p0+T)/4;
    float total[Tokens][2]={};
    for(int head0=0;head0<64;head0+=8){
        if constexpr(Scaled){
            // One exact power-of-two normalization per query/head. Keep the
            // exponent as an integer: even subnormal FP32 queries need no
            // overflowing floating-point scale factor. All lanes participate.
            for(int i=warp;i<Tokens*8;i+=4){
                float maximum=0;
                for(int d=lane;d<128;d+=32){
                    const float x=t0+i/8<T?iq[((size_t)(t0+i/8)*64+head0+i%8)*128+d]:0;
                    maximum=fmaxf(maximum,fabsf(x));
                }
                for(int offset=16;offset;offset>>=1)
                    maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
                if(lane==0){
                    int exponent=0;if(maximum>0)frexpf(maximum,&exponent);
                    const int shift=-exponent;shifts[i]=shift;
                    // Both factors are normal and exact in this common range.
                    // Avoid a libdevice scalbn operation for every query value
                    // and accumulator; extreme exponents retain that full path.
                    const bool fast=shift>=-126 && shift<=126;
                    scales[i]=fast?__uint_as_float(unsigned(127+shift)<<23):0;
                    inverse_scales[i]=fast?__uint_as_float(unsigned(127-shift)<<23):0;
                }
            }
            __syncthreads();
        }
        for(int i=threadIdx.x;i<Tokens*8*128;i+=128){
            const int t=i/(8*128),h=(i/128)%8,d=i%128;
            const int dst=(t*8+h)*QStride+d;
            if constexpr(Packed){
                const uint32_t value=t0+t<T?packed[((size_t)(t0+t)*64+head0+h)*128+d]:0;
                qhi[dst]=uint16_t(value);qlo[dst]=uint16_t(value>>16);
            }else{
            float q=t0+t<T?iq[((size_t)(t0+t)*64+head0+h)*128+d]:0;
            if constexpr(Scaled){
                const float factor=scales[t*8+h];
                q=factor!=0?__fmul_rn(q,factor):scalbnf(q,shifts[t*8+h]);
            }
            const __half hi=__float2half_rn(q);
            qhi[dst]=__half_as_ushort(hi);
            qlo[dst]=__half_as_ushort(__float2half_rn(q-__half2float(hi)));
            }
        }
        for(int i=threadIdx.x;i<Tokens*8;i+=128)
            w[i]=t0+i/8<T?weights[(size_t)(t0+i/8)*64+head0+i%8]:0;
        __syncthreads();
        float hi[Tokens][8]={},lo[Tokens][8]={};
        for(int k0=0;k0<128;k0+=4){
            uint32_t a[2]={};
#pragma unroll
            for(int k=0;k<4;++k){
                const uint16_t key=row0+j<max_vis?keys[(size_t)(row0+j)*128+k0+k]:0;
                a[k>>1]|=uint32_t(key)<<(16*(k&1));
            }
            // Reuse A's bit carriers in registers; all 32 lanes execute every
            // MMA, including masked key rows and query tails.
#pragma unroll
            for(int t=0;t<Tokens;++t){
                uint32_t bh[2]={},bl[2]={};
#pragma unroll
                for(int k=0;k<4;++k){
                    bh[k>>1]|=uint32_t(qhi[(t*8+j)*QStride+k0+k])<<(16*(k&1));
                    bl[k>>1]|=uint32_t(qlo[(t*8+j)*QStride+k0+k])<<(16*(k&1));
                }
                mma884(hi[t],a[0],a[1],bh[0],bh[1]);
                mma884(lo[t],a[0],a[1],bl[0],bl[1]);
            }
        }
#pragma unroll
        for(int t=0;t<Tokens;++t){
#pragma unroll
            for(int r=0;r<2;++r){
                float sum=0;
#pragma unroll
                for(int band=0;band<2;++band){
#pragma unroll
                    for(int c=0;c<2;++c){
                        const int i=2*r+4*band+c;
                        const int h=(lane&2)+4*band+c;
                        float dot=hi[t][i]+lo[t][i];
                        if constexpr(Scaled){
                            const float factor=inverse_scales[t*8+h];
                            dot=factor!=0?__fmul_rn(dot,factor):scalbnf(dot,-shifts[t*8+h]);
                        }
                        sum+=fmaxf(dot,0.0f)*w[t*8+h];
                    }
                }
                sum+=__shfl_xor_sync(0xffffffffu,sum,2);
                total[t][r]+=sum;
            }
        }
        __syncthreads(); // no warp overwrites the next band's queries early
    }
    if(!(lane&2)){
#pragma unroll
        for(int t=0;t<Tokens;++t){
#pragma unroll
            for(int r=0;r<2;++r){
                const int row=row0+(lane&1)+4*(lane>>4)+2*r;
                if(t0+t<T && row<(p0+t0+t+1)/4){
                    float value=total[t][r];
                    if constexpr(Packed){
                        const int shift=packed_shifts[t0+t];
                        value=shift>=-126&&shift<=126?__fmul_rn(value,__uint_as_float(unsigned(127-shift)<<23)):scalbnf(value,-shift);
                    }
                    score[(size_t)(t0+t)*ld+row]=value;
                }
            }
        }
    }
#else
    (void)iq;(void)keys;(void)weights;(void)p0;(void)T;(void)ld;(void)score;
    (void)packed;(void)packed_shifts;
#endif
}
}
#endif
