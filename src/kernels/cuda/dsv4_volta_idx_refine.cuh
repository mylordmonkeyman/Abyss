#pragma once
// Isolated research candidate only. The relative cutoff band is empirical,
// not a certified global error bound; no production/default admission here.
#ifndef STRATA_USE_HIP
#include "dsv4_volta_idx_runner.cuh"
#include "dsa_topk.cuh"
namespace strata::kernels::volta {
constexpr int kRefineCapacity=8192,kRefinePad=129;
// Literal upstream idx_select_batch_kernel body and its shared selector helper.
// Launch once per pass, rather than T single-CTA launches per pass.
__global__ void __launch_bounds__(1024) select_refinement_batch(const float* score,int ld,int p0,int* selected){
    const int t=blockIdx.x,n=(p0+t+1)/4;
    if(n<=0)return;
    strata::kernels::dsa::select_top_pools(score+(size_t)t*ld,n,min(512,n),1,selected+(size_t)t*512);
}
inline void select_refinement(const float* score,int ld,int p0,int T,int* selected,cudaStream_t stream){
    select_refinement_batch<<<T,1024,0,stream>>>(score,ld,p0,selected);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("refined MMA batched selection failed");
}
__global__ void __launch_bounds__(256) refine_scaled_selection(
    const float* q,const uint16_t* keys,const float* weights,int p0,int T,int ld,
    float* score,const int* selected,uint32_t* work,int* counts){
    const int t=blockIdx.x,tid=threadIdx.x,h=tid%64,group=tid/64;
    if(t>=T)return;
    const int n=(p0+t+1)/4,top=min(512,n);
    __shared__ float peaks[8],threshold;
    __shared__ int count,full;
    __shared__ float query[64*kRefinePad],key[4*kRefinePad],dot[256];
    float peak=0;
    for(int j=tid;j<n;j+=256)peak=fmaxf(peak,fabsf(score[(size_t)t*ld+j]));
    for(int d=16;d;d>>=1)peak=fmaxf(peak,__shfl_down_sync(0xffffffffu,peak,d));
    if((tid&31)==0)peaks[tid/32]=peak;
    __syncthreads();
    if(tid==0){
        float maximum=0;for(int i=0;i<8;++i)maximum=fmaxf(maximum,peaks[i]);
        // Include every approximate top-512 entry and an additional cutoff
        // band. Full rescore on compact-list overflow, never truncate rows.
        threshold=score[(size_t)t*ld+selected[(size_t)t*512+top-1]]-maximum*1e-4f;
        count=0;
    }
    __syncthreads();
    uint32_t* list=work+(size_t)t*kRefineCapacity;
    for(int base=0;base<n;base+=256){
        const int j=base+tid;
        const unsigned mask=__ballot_sync(0xffffffffu,j<n && score[(size_t)t*ld+j]>=threshold);
        const int lane=tid&31;int offset=0;
        if(lane==0)offset=atomicAdd(&count,__popc(mask));
        offset=__shfl_sync(0xffffffffu,offset,0);
        if(mask&(1u<<lane)){
            const int rank=offset+__popc(mask&((1u<<lane)-1));
            if(rank<kRefineCapacity)list[rank]=j;
        }
    }
    __syncthreads();
    if(tid==0){full=count>kRefineCapacity;counts[t]=full?n:count;}
    for(int i=tid;i<64*128;i+=256)query[(i/128)*kRefinePad+i%128]=q[(size_t)t*64*128+i];
    __syncthreads();
    const int total=full?n:count;
    for(int base=0;base<total;base+=4){
        const int index=base+group,j=index<total?(full?index:int(list[index])):0;
        for(int d=h;d<128;d+=64)
            key[group*kRefinePad+d]=index<total?__half2float(__ushort_as_half(keys[(size_t)j*128+d])):0;
        __syncthreads();
        float value=0;
        // Preserve the operational baseline's sequential 128-D FMA and then
        // sequential 64-head FMA, including independent per-head ReLU.
        for(int d=0;d<128;++d)value=__fmaf_rn(query[h*kRefinePad+d],key[group*kRefinePad+d],value);
        dot[tid]=value;
        __syncthreads();
        if(h==0 && index<total){
            float result=0;
            for(int head=0;head<64;++head)result=__fmaf_rn(fmaxf(dot[group*64+head],0.0f),weights[(size_t)t*64+head],result);
            score[(size_t)t*ld+j]=result;
        }
        __syncthreads();
    }
}

inline void indexer_batch_scaled_refined(const float* q,const uint16_t* k,const float* w,
    int p0,int T,int ld,float* scores,int* ids,cudaStream_t stream,
    uint32_t* packed,size_t packed_count,int* counts,size_t count_count){
    if((p0+1)/4<=0)throw std::runtime_error("refined MMA requires at least one visible key per query");
    indexer_batch_scaled(q,k,w,p0,T,ld,scores,ids,stream,packed,packed_count,counts,count_count,false);
    select_refinement(scores,ld,p0,T,ids,stream);
    // Packed hi/lo and exponents are dead after the first score/select. Reuse
    // them as per-token compact row lists/counts, with no new allocation.
    refine_scaled_selection<<<T,256,0,stream>>>(q,k,w,p0,T,ld,scores,ids,packed,counts);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("scaled MMA F32 rescore launch failed");
    select_refinement(scores,ld,p0,T,ids,stream);
}
}
#endif
