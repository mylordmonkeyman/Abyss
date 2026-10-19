// Shared isolated SM70 diagnostic arithmetic; no production dispatch.
#pragma once
#ifndef STRATA_USE_HIP
#include "volta_mma884.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

namespace v100diag {
constexpr int H=64,D=512,GUARD=32;
constexpr float sentinel=12345.0f;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void need(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(const std::vector<T>& v):n(v.size()) {
        ck(cudaMalloc((void**)&p,n*sizeof(T)));
        ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Buffer(){cudaFree(p);}
    std::vector<T> get() const {
        std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;
    }
};
__device__ __forceinline__ void split(float value,uint16_t& hi,uint16_t& lo) {
    const __half a=__float2half_rn(value);
    hi=__half_as_ushort(a);lo=__half_as_ushort(__float2half_rn(value-__half2float(a)));
}
__global__ void qk_mma(const float* q,const uint16_t* keys,float* scores,int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700 && __CUDA_ARCH__<750
    const int lane=threadIdx.x,qp=(lane>>2)&3,j=(lane&3)+4*(lane>>4);
    const int head=blockIdx.x*8,key=blockIdx.y*32+qp*8;
    float hi[8]={},lo[8]={};
    for(int k0=0;k0<D;k0+=4) {
        uint32_t ah[2]={},al[2]={},b[2]={};
#pragma unroll
        for(int k=0;k<4;++k) {
            uint16_t a0,a1;split(q[(head+j)*D+k0+k],a0,a1);
            const uint16_t v=key+j<n?keys[(key+j)*D+k0+k]:0;
            ah[k/2]|=uint32_t(a0)<<(16*(k%2));al[k/2]|=uint32_t(a1)<<(16*(k%2));
            b[k/2]|=uint32_t(v)<<(16*(k%2));
        }
        strata::kernels::volta::mma884(hi,ah[0],ah[1],b[0],b[1]);
        strata::kernels::volta::mma884(lo,al[0],al[1],b[0],b[1]);
    }
#pragma unroll
    for(int i=0;i<8;++i) {
        const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1),c=(lane&2)+4*(i>>2)+(i&1);
        if(key+c<n) scores[(head+r)*n+key+c]=hi[i]+lo[i];
    }
#endif
}
// Diagnostic alternative: shorter independent hi/lo chains, then compensated
// FP32 combination at explicit 64-dimension boundaries. Not production dispatch.
__global__ void qk_mma_chunked(const float* q,const uint16_t* keys,float* scores,int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700 && __CUDA_ARCH__<750
    const int lane=threadIdx.x,qp=(lane>>2)&3,j=(lane&3)+4*(lane>>4);
    const int head=blockIdx.x*8,key=blockIdx.y*32+qp*8;
    float total[8]={},correction[8]={};
    for(int begin=0;begin<D;begin+=64) {
      float hi[8]={},lo[8]={};
      for(int k0=begin;k0<begin+64;k0+=4) {
        uint32_t ah[2]={},al[2]={},b[2]={};
#pragma unroll
        for(int k=0;k<4;++k) {
            uint16_t a0,a1;split(q[(head+j)*D+k0+k],a0,a1);
            const uint16_t v=key+j<n?keys[(key+j)*D+k0+k]:0;
            ah[k/2]|=uint32_t(a0)<<(16*(k%2));al[k/2]|=uint32_t(a1)<<(16*(k%2));
            b[k/2]|=uint32_t(v)<<(16*(k%2));
        }
        strata::kernels::volta::mma884(hi,ah[0],ah[1],b[0],b[1]);
        strata::kernels::volta::mma884(lo,al[0],al[1],b[0],b[1]);
      }
#pragma unroll
      for(int i=0;i<8;++i) {
          const float y=(hi[i]+lo[i])-correction[i],next=total[i]+y;
          correction[i]=(next-total[i])-y;total[i]=next;
      }
    }
#pragma unroll
    for(int i=0;i<8;++i) {
        const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1),c=(lane&2)+4*(i>>2)+(i&1);
        if(key+c<n) scores[(head+r)*n+key+c]=total[i];
    }
#endif
}
// Deliberately simple staged reference-normalization kernel. Full score scratch
// and serial head reduction are diagnostic, not a fused attention design.
__global__ void softmax_sink(const float* score,const float* sinks,float* p,float* den,int n) {
    const int h=threadIdx.x;float m=sinks[h];const float scale=1.0f/sqrtf(float(D));
    for(int k=0;k<n;++k) m=fmaxf(m,score[h*n+k]*scale);
    float d=expf(sinks[h]-m);
    for(int k=0;k<n;++k) d+=expf(score[h*n+k]*scale-m);
    den[h]=d;
    for(int k=0;k<n;++k) p[h*n+k]=expf(score[h*n+k]*scale-m)/d;
}
__global__ void pv_mma(const float* p,const uint16_t* keys,float* out,int n,float pscale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700 && __CUDA_ARCH__<750
    const int lane=threadIdx.x,qp=(lane>>2)&3,j=(lane&3)+4*(lane>>4);
    const int head=blockIdx.x*8,dim=blockIdx.y*32+qp*8;
    float hi[8]={},lo[8]={};
    for(int k0=0;k0<n;k0+=4) {
        uint32_t ah[2]={},al[2]={},b[2]={};
#pragma unroll
        for(int k=0;k<4;++k) {
            uint16_t a0,a1;split(k0+k<n?p[(head+j)*n+k0+k]*pscale:0.0f,a0,a1);
            const uint16_t v=k0+k<n?keys[(k0+k)*D+dim+j]:0;
            ah[k/2]|=uint32_t(a0)<<(16*(k%2));al[k/2]|=uint32_t(a1)<<(16*(k%2));
            b[k/2]|=uint32_t(v)<<(16*(k%2));
        }
        strata::kernels::volta::mma884(hi,ah[0],ah[1],b[0],b[1]);
        strata::kernels::volta::mma884(lo,al[0],al[1],b[0],b[1]);
    }
#pragma unroll
    for(int i=0;i<8;++i) {
        const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1),c=(lane&2)+4*(i>>2)+(i&1);
        out[(head+r)*D+dim+c]=(hi[i]+lo[i])/pscale;
    }
#endif
}
__global__ void qk_scalar(const float* q,const uint16_t* keys,float* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*n)return;
    float v=0;for(int k=0;k<D;++k) v=fmaf(q[(i/n)*D+k],__half2float(__ushort_as_half(keys[(i%n)*D+k])),v);
    out[i]=v;
}
__global__ void pv_scalar(const float* p,const uint16_t* keys,float* out,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H*D)return;
    float v=0;for(int k=0;k<n;++k)v=fmaf(p[(i/D)*n+k],__half2float(__ushort_as_half(keys[k*D+i%D])),v);
    out[i]=v;
}
struct Error { double rel,max_abs,p99_abs; };
Error compare(const std::vector<float>& got,const std::vector<double>& ref) {
    double error=0,norm=0,mx=0;std::vector<double> absolute;
    for(size_t i=0;i<ref.size();++i) {
        need(std::isfinite(got[i]),"nonfinite arithmetic output");
        const double e=got[i]-ref[i];error+=e*e;norm+=ref[i]*ref[i];mx=std::max(mx,std::abs(e));
        absolute.push_back(std::abs(e));
    }
    for(size_t i=ref.size();i<got.size();++i)need(got[i]==sentinel,"output guard overwritten");
    need(norm>0 || error==0,"nonzero output for exact zero reference");
    std::sort(absolute.begin(),absolute.end());
    const double p99=absolute.empty()?0:absolute[size_t(.99*double(absolute.size()-1))];
    return {norm>0?std::sqrt(error/norm):0,mx,p99};
}
} // namespace v100diag
#endif
