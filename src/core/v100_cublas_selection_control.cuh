#pragma once
// Diagnostic-only intervention: retain cuBLAS work but substitute F32 decisions.
// The caller excludes its extra scoring, copies and synchronization from timing.
#ifndef STRATA_USE_HIP
#include "strata/kernels/dsv4_fast.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::core::v100diag {
__global__ void selection_control_compare(const float* candidate_score,int ld,int p0,
    int* selected,const int* prior,int* counts,bool membership_only){
    const int t=blockIdx.x,i=threadIdx.x,top=min(512,(p0+t+1)/4);
    __shared__ int reference[512],original[512];
    if(i<top){reference[i]=selected[(size_t)t*512+i];original[i]=prior[(size_t)t*512+i];}
    __syncthreads();
    bool found=false;
    if(i<top)for(int j=0;j<top;++j)found|=reference[i]==original[j];
    const int overlap=__syncthreads_count(i<top && found);
    const int differences=__syncthreads_count(i<top && reference[i]!=original[i]);
    if(i==0){atomicAdd(counts,1);atomicAdd(counts+1,differences==0);atomicAdd(counts+2,overlap==top);atomicAdd(counts+3,overlap);}
    // Preserve reference membership but order it by the original cuBLAS scores.
    // Rank ties by original row ID, matching the unchanged selector contract.
    if(membership_only && i<top){
        const int id=reference[i];const float value=candidate_score[(size_t)t*ld+id];int rank=0;
        for(int j=0;j<top;++j){const int other=reference[j];const float score=candidate_score[(size_t)t*ld+other];
            rank+=score>value || (score==value && other<id);}
        selected[(size_t)t*512+rank]=id;
    }
}
inline void cublas_selection_control(float* q,const uint16_t* keys,const float* weights,
    int p0,int T,int ld,const float* candidate_score,int* selected,cudaStream_t stream,
    float* reference_score,size_t reference_count,int* scratch,size_t scratch_count,bool membership_only){
    const int visible=(p0+T)/4;
    if(T<=0 || (p0+1)/4<=0 || ld<visible || reference_count<size_t(T)*visible || scratch_count<size_t(T)*512+4)
        throw std::runtime_error("cuBLAS selection-control scratch/shape contract violated");
    auto ck=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    int* counts=scratch+(size_t)T*512;
    ck(cudaMemcpyAsync(scratch,selected,(size_t)T*512*sizeof(int),cudaMemcpyDeviceToDevice,stream));
    ck(cudaMemsetAsync(counts,0,4*sizeof(int),stream));
    // Use the operational batch reference, not a mathematically different dot.
    strata::kernels::dsv4::indexer_batch(q,keys,weights,64,4,p0,T,512,reference_score,visible,selected,stream);
    selection_control_compare<<<T,512,0,stream>>>(candidate_score,ld,p0,selected,scratch,counts,membership_only);
    ck(cudaGetLastError());
}
}
#endif
