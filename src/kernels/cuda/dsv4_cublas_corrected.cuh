#pragma once
// Isolated correction candidates. No production/runtime dispatch from this file.
#ifndef STRATA_USE_HIP
#include "dsv4_cublas_idx.cuh"
#include "dsv4_volta_idx_refine.cuh"
namespace strata::kernels::dsv4::p1::correction {
enum class Policy { none,full_flagged,band_flagged,band_all };
constexpr int Capacity=8192,StateWords=516,Pad=129;
// Calibrated on captured rows; empirical, not a certified global error bound.
constexpr float RelativeEnvelope=32.0f*0x1p-23f;
__global__ void sequential_epilogue(const float* dots,const float* weights,float* scores,
    int p0,int t0,int tokens,int j0,int rows,int ld){
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,item=blockIdx.x*8+warp;
    if(item>=tokens*rows)return;
    const int t=item/rows,j=item%rows;
    if(j0+j>=(p0+t0+t+1)/4)return;
    __shared__ float values[8*64];
    for(int h=lane;h<64;h+=32)values[warp*64+h]=dots[(size_t)j*tokens*64+t*64+h];
    __syncwarp();
    if(lane==0){float sum=0;
        for(int h=0;h<64;++h)sum=__fmaf_rn(fmaxf(values[warp*64+h],0.0f),weights[(t0+t)*64+h],sum);
        scores[(size_t)(t0+t)*ld+j0+j]=sum;
    }
}
inline void scores(cublasHandle_t handle,cudaStream_t stream,const float* q,const uint16_t* k,const float* w,
    int p0,int T,int ld,float* out,float* key,size_t key_count,float* dot,size_t dot_count,bool sequential){
    if(T<=0||p0<0||ld<(p0+T)/4||key_count<size_t(KK)*DIM||dot_count<size_t(std::min(T,TT))*HEADS*KK)
        throw std::runtime_error("corrected cuBLAS score scratch/shape contract violated");
    const float one=1,zero=0;
    // Same key-major/tile geometry and FP32 SGEMM arguments as the original.
    for(int j0=0;j0<(p0+T)/4;j0+=KK){
        const int rows=std::min(KK,(p0+T)/4-j0);
        unpack<<<(rows*DIM+255)/256,256,0,stream>>>(k+(size_t)j0*DIM,key,rows);ck(cudaGetLastError());
        for(int t0=0;t0<T;t0+=TT){const int tokens=std::min(TT,T-t0);
            if(j0>=(p0+t0+tokens)/4)continue;
            blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,tokens*HEADS,rows,DIM,&one,
                q+(size_t)t0*HEADS*DIM,DIM,key,DIM,&zero,dot,tokens*HEADS));
            if(sequential)sequential_epilogue<<<(tokens*rows+7)/8,256,0,stream>>>(dot,w,out,p0,t0,tokens,j0,rows,ld);
            else epilogue<<<(tokens*rows+7)/8,256,0,stream>>>(dot,w,out,p0,t0,tokens,j0,rows,ld);
            ck(cudaGetLastError());
        }
    }
}
__global__ void __launch_bounds__(1024) top513(const float* score,int ld,int p0,int* ids){
    const int t=blockIdx.x,n=(p0+t+1)/4;
    strata::kernels::dsa::select_top_pools(score+(size_t)t*ld,n,min(513,n),1,ids+(size_t)t*513);
}
__global__ void detect(const float* score,int ld,int p0,const int* top,int* ids,int* flags,float* thresholds){
    const int t=blockIdx.x,tid=threadIdx.x,n=(p0+t+1)/4,k=min(512,n);
    __shared__ float peaks[8];float maximum=0;
    for(int j=tid;j<n;j+=256)maximum=fmaxf(maximum,fabsf(score[(size_t)t*ld+j]));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    if(!(tid&31))peaks[tid/32]=maximum;
    __syncthreads();
    if(tid==0){maximum=0;for(int i=0;i<8;++i)maximum=fmaxf(maximum,peaks[i]);
        const float envelope=maximum*RelativeEnvelope,cut=score[(size_t)t*ld+top[(size_t)t*513+k-1]];
        const float margin=n>512?cut-score[(size_t)t*ld+top[(size_t)t*513+512]]:0;
        flags[t]=n<=512 || margin<=2*envelope;thresholds[t]=cut-2*envelope;
    }
    for(int i=tid;i<k;i+=256)ids[(size_t)t*512+i]=top[(size_t)t*513+i];
}
template<bool Full,bool All>__global__ void __launch_bounds__(256) rescore(const float* q,const uint16_t* keys,const float* weights,
    int p0,int ld,float* score,const int* flags,const float* thresholds,uint32_t* work,int* counts){
    const int t=blockIdx.x,tid=threadIdx.x,h=tid%64,group=tid/64,n=(p0+t+1)/4;
    if(!All && !flags[t]){if(tid==0)counts[t]=0;return;}
    __shared__ int count,full;
    __shared__ float query[64*Pad],key[4*Pad],dots[256];
    if(tid==0)count=Full?n:0;
    __syncthreads();uint32_t* list=work+(size_t)t*Capacity;
    if(!Full)for(int base=0;base<n;base+=256){
        const int j=base+tid,lane=tid&31;
        const unsigned mask=__ballot_sync(0xffffffffu,j<n && score[(size_t)t*ld+j]>=thresholds[t]);
        int offset=0;if(lane==0)offset=atomicAdd(&count,__popc(mask));offset=__shfl_sync(0xffffffffu,offset,0);
        if(mask&(1u<<lane)){const int rank=offset+__popc(mask&((1u<<lane)-1));if(rank<Capacity)list[rank]=j;}
    }
    __syncthreads();if(tid==0){full=Full||count>Capacity;counts[t]=full?n:count;}
    for(int i=tid;i<64*128;i+=256)query[(i/128)*Pad+i%128]=q[(size_t)t*64*128+i];
    __syncthreads();const int total=full?n:count;
    for(int base=0;base<total;base+=4){
        const int index=base+group,j=index<total?(full?index:int(list[index])):0;
        for(int d=h;d<128;d+=64)key[group*Pad+d]=index<total?__half2float(__ushort_as_half(keys[(size_t)j*128+d])):0;
        __syncthreads();float value=0;
        // Match operational F32 dimension order and weighted head FMA order.
        for(int d=0;d<128;++d)value=__fmaf_rn(query[h*Pad+d],key[group*Pad+d],value);
        dots[tid]=value;__syncthreads();
        if(h==0&&index<total){float result=0;
            for(int head=0;head<64;++head)result=__fmaf_rn(fmaxf(dots[group*64+head],0.0f),weights[(size_t)t*64+head],result);
            score[(size_t)t*ld+j]=result;
        }
        __syncthreads();
    }
}
inline void run(cublasHandle_t handle,cudaStream_t stream,const float* q,const uint16_t* keys,const float* weights,
    int p0,int T,int ld,float* score,int* selected,float* key,size_t key_count,float* dot,size_t dot_count,
    uint32_t* work,size_t work_count,int* state,size_t state_count,Policy policy,bool sequential){
    if((p0+1)/4<=0||work_count<size_t(T)*Capacity||state_count<size_t(T)*StateWords)
        throw std::runtime_error("cuBLAS correction scratch/visibility contract violated");
    scores(handle,stream,q,keys,weights,p0,T,ld,score,key,key_count,dot,dot_count,sequential);
    int* counts=state;int* flags=counts+T;int* top=flags+T;float* thresholds=reinterpret_cast<float*>(top+(size_t)T*513);
    if(policy==Policy::none){
        ck(cudaMemsetAsync(state,0,2*T*sizeof(int),stream));
        strata::kernels::volta::select_refinement(score,ld,p0,T,selected,stream);return;
    }
    top513<<<T,1024,0,stream>>>(score,ld,p0,top);ck(cudaGetLastError());
    detect<<<T,256,0,stream>>>(score,ld,p0,top,selected,flags,thresholds);ck(cudaGetLastError());
    if(policy==Policy::full_flagged)rescore<true,false><<<T,256,0,stream>>>(q,keys,weights,p0,ld,score,flags,thresholds,work,counts);
    else if(policy==Policy::band_flagged)rescore<false,false><<<T,256,0,stream>>>(q,keys,weights,p0,ld,score,flags,thresholds,work,counts);
    else rescore<false,true><<<T,256,0,stream>>>(q,keys,weights,p0,ld,score,flags,thresholds,work,counts);
    ck(cudaGetLastError());
    strata::kernels::volta::select_refinement(score,ld,p0,T,selected,stream);
}
}
#endif
