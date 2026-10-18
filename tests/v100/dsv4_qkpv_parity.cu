// This target is CUDA-only in CMake; exclude it from HIP source auditing.
#ifndef STRATA_USE_HIP
// P2a arithmetic only: 64 heads, shared 512-D FP16 K=V, independent hi/lo
// accumulators, sink denominator with no sink value. No model dispatch/gathers.
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

namespace {
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
void run(int n,int mode,float pscale) {
    std::mt19937 rng(20261018+n+mode);std::uniform_real_distribution<float> uniform(-1.f,1.f);
    std::vector<float> q(H*D),sinks(H);
    std::vector<uint16_t> keys(std::max(1,n*D));
    for(float& v:q)v=mode==3?0:uniform(rng)*(mode==4?8.0f:1.0f);
    for(auto& v:keys)v=__half_as_ushort(__float2half_rn(uniform(rng)*.125f));
    for(int h=0;h<H;++h)sinks[h]=mode==1?80.f:mode==2?-80.f:uniform(rng);
    Buffer<float> dq(q),dsink(sinks),dscore(std::vector<float>(H*n+GUARD,sentinel)),
                  dscalar(std::vector<float>(H*n+GUARD,sentinel)),dp(std::vector<float>(H*n+GUARD,sentinel)),
                  dden(std::vector<float>(H+GUARD,sentinel)),dout(std::vector<float>(H*D+GUARD,sentinel)),
                  dpvscalar(std::vector<float>(H*D+GUARD,sentinel));
    Buffer<uint16_t> dk(keys);
    if(n) {
        qk_mma<<<dim3(H/8,(n+31)/32),32>>>(dq.p,dk.p,dscore.p,n);ck(cudaGetLastError());
        qk_scalar<<<(H*n+127)/128,128>>>(dq.p,dk.p,dscalar.p,n);ck(cudaGetLastError());
    }
    softmax_sink<<<1,H>>>(dscore.p,dsink.p,dp.p,dden.p,n);ck(cudaGetLastError());
    pv_mma<<<dim3(H/8,D/32),32>>>(dp.p,dk.p,dout.p,n,pscale);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    if(n)qk_mma<<<dim3(H/8,(n+31)/32),32,0,stream>>>(dq.p,dk.p,dscore.p,n);
    softmax_sink<<<1,H,0,stream>>>(dscore.p,dsink.p,dp.p,dden.p,n);
    pv_mma<<<dim3(H/8,D/32),32,0,stream>>>(dp.p,dk.p,dout.p,n,pscale);
    ck(cudaGetLastError());ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int i=0;i<4;++i)ck(cudaGraphLaunch(executable,stream));
    ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    pv_scalar<<<(H*D+127)/128,128>>>(dp.p,dk.p,dpvscalar.p,n);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<double> score(H*n),prob(H*n),den(H),out(H*D);
    for(int h=0;h<H;++h) {
        double mx=sinks[h];
        for(int r=0;r<n;++r) {
            double dot=0;for(int d=0;d<D;++d)dot+=double(q[h*D+d])*__half2float(__ushort_as_half(keys[r*D+d]));
            score[h*n+r]=dot;mx=std::max(mx,dot/std::sqrt(double(D)));
        }
        den[h]=std::exp(sinks[h]-mx);
        for(int r=0;r<n;++r)den[h]+=std::exp(score[h*n+r]/std::sqrt(double(D))-mx);
        for(int r=0;r<n;++r)prob[h*n+r]=std::exp(score[h*n+r]/std::sqrt(double(D))-mx)/den[h];
        for(int d=0;d<D;++d)for(int r=0;r<n;++r)
            out[h*D+d]+=prob[h*n+r]*__half2float(__ushort_as_half(keys[r*D+d]));
    }
    const Error eq=compare(dscore.get(),score),ep=compare(dp.get(),prob),ed=compare(dden.get(),den),eo=compare(dout.get(),out);
    const auto scalar_pv=dpvscalar.get();compare(scalar_pv,out);
    const std::vector<double> scalar_pv_reference(scalar_pv.begin(),scalar_pv.begin()+H*D);
    const Error em=compare(dout.get(),scalar_pv_reference);
    if(n) { const Error scalar=compare(dscalar.get(),score);need(scalar.rel<1e-5,"scalar QK reference error"); }
    need(eq.rel<1e-5 && ep.rel<1e-5 && ed.rel<1e-5,"QK/sink-softmax component screen");
    need(eo.rel<(pscale==1?1e-4:1e-5) || eo.max_abs<1e-10,"PV component screen");
    need(em.rel<(pscale==1?1e-4:1e-5) || em.max_abs<1e-10,"PV scalar comparison");
    std::printf("{\"case\":\"contiguous_qkpv\",\"heads\":64,\"latent_dim\":512,\"keys\":%d,\"mode\":%d,"
                "\"probability_power_of_two_scale\":%.0f,\"qk_relative_l2\":%.9g,\"prob_relative_l2\":%.9g,"
                "\"sink_den_relative_l2\":%.9g,\"pv_relative_l2\":%.9g,\"pv_max_abs\":%.9g,"
                "\"pv_vs_scalar_relative_l2\":%.9g,\"qk_max_abs\":%.9g,\"qk_p99_abs\":%.9g,"
                "\"prob_max_abs\":%.9g,\"sink_den_max_abs\":%.9g,\"pv_p99_abs\":%.9g,\"pass\":true}\n",
                n,mode,pscale,eq.rel,ep.rel,ed.rel,eo.rel,eo.max_abs,em.rel,
                eq.max_abs,eq.p99_abs,ep.max_abs,ed.max_abs,eo.p99_abs);
}
}
int main() {
    try {
        cudaDeviceProp device{};ck(cudaGetDeviceProperties(&device,0));
        need(device.major==7 && device.minor==0,"QK/PV diagnostic requires actual SM70");
        for(int n:{0,1,7,8,15,16,31,32,33,127,128,129,512,640}) {
            run(n,0,1);run(n,0,1024);
        }
        for(int mode:{1,2,3,4})run(129,mode,1024);
        cudaFuncAttributes qk{},pv{};ck(cudaFuncGetAttributes(&qk,qk_mma));ck(cudaFuncGetAttributes(&pv,pv_mma));
        int qk_blocks=0,pv_blocks=0;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qk_blocks,qk_mma,32,0));
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pv_blocks,pv_mma,32,0));
        std::printf("{\"suite\":\"contiguous_qkpv\",\"cases\":32,\"qk_registers\":%d,\"pv_registers\":%d,"
                    "\"qk_local_bytes\":%zu,\"pv_local_bytes\":%zu,\"theoretical_qk_blocks_per_sm\":%d,"
                    "\"theoretical_pv_blocks_per_sm\":%d,\"graph_replays_per_case\":4,"
                    "\"production_enabled\":false,\"pass\":true}\n",
                    qk.numRegs,pv.numRegs,qk.localSizeBytes,pv.localSizeBytes,qk_blocks,pv_blocks);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"QK/PV diagnostic FAILED: %s\n",e.what());return 1;}
}

#endif // !STRATA_USE_HIP
