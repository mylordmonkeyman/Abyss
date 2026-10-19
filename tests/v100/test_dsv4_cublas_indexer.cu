// P1e component test of the shared experimental runtime scorer.
#ifndef STRATA_USE_HIP
#include "../../src/kernels/cuda/dsv4_cublas_idx.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace ds = strata::kernels::dsv4;
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void blas(cublasStatus_t e) { if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS call failed"); }
void need(bool b,const char* s) {if(!b)throw std::runtime_error(s);}
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t count):n(count) {ck(cudaMalloc((void**)&p,n*sizeof(T)));}
    explicit Buffer(const std::vector<T>& x):Buffer(x.size()) {ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(p);} Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    std::vector<T> get()const{std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
constexpr int TT=256, KK=512, HEADS=64, DIM=128, ROWS_PER_WARP=1;
struct Scorer {
    // Scratch is capped independently of the common caller-owned score/selection buffers.
    static constexpr size_t WS=16*1024*1024;
    static_assert(WS+(size_t(KK)*DIM+size_t(TT)*HEADS*KK)*sizeof(float)<64*1024*1024,
        "caller-owned cuBLAS scratch cap");
    Buffer<float> key{size_t(KK)*DIM},dot{size_t(TT)*HEADS*KK};Buffer<unsigned char> workspace{WS};
    cublasHandle_t handle{};
    Scorer(){blas(cublasCreate(&handle));blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
        blas(cublasSetWorkspace(handle,workspace.p,WS));}
    ~Scorer(){cublasDestroy(handle);}
    void run(const float* q,const uint16_t* k,const float* w,int p0,int T,int ld,float* score,int* ids){
        ds::p1::run(handle,nullptr,q,k,w,p0,T,ld,score,ids,key.p,key.n,dot.p,dot.n);
    }
};
uint32_t rng;
float random_value(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (int(rng%65537)-32768)/32768.0f;}
template<class F> float timing(F f,int repeats){
    f();ck(cudaDeviceSynchronize());cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a));for(int i=0;i<repeats;++i)f();ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));
    float ms=0;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/repeats;
}
void run_case(Scorer& scorer,int T,int p0,int mode,bool bench){
    rng=0x823947u^uint32_t(T)^uint32_t(p0)^(uint32_t(mode)<<20);
    const int visible=(p0+T)/4,ld=visible+7;
    std::vector<float> q(size_t(T)*HEADS*DIM),w(T*HEADS),blank(size_t(T)*ld+32,12345.0f);
    std::vector<uint16_t> k(size_t(visible)*DIM);
    for(auto& x:q)x=random_value();for(auto& x:w)x=random_value();
    for(auto& x:k)x=__half_as_ushort(__float2half_rn(random_value()));
    if(mode==1)std::fill(q.begin(),q.end(),0);
    if(mode==2)for(auto& x:q)x*=std::ldexp(1.0f,-20);
    if(mode==3)for(int j=1;j<visible;++j)if(j&1)std::copy_n(k.begin()+size_t(j-1)*DIM,DIM,k.begin()+size_t(j)*DIM);
    Buffer<float> dq(q),dw(w),out(blank),old(blank);Buffer<uint16_t> dk(k);
    Buffer<int> ids(std::vector<int>(size_t(T)*512+32,-12345)),baseline(std::vector<int>(size_t(T)*512+32,-12345));
    auto oldrun=[&]{ds::indexer_batch(dq.p,dk.p,dw.p,HEADS,4,p0,T,512,old.p,ld,baseline.p,nullptr);};
    auto newrun=[&]{scorer.run(dq.p,dk.p,dw.p,p0,T,ld,out.p,ids.p);};
    oldrun();newrun();ck(cudaDeviceSynchronize());
    if(!bench){
        cudaStream_t stream;ck(cudaStreamCreate(&stream));
        // Set stream before workspace: cuBLAS resets workspace when stream changes.
        blas(cublasSetStream(scorer.handle,stream));
        blas(cublasSetWorkspace(scorer.handle,scorer.workspace.p,Scorer::WS));
        auto launch=[&]{ds::p1::run(scorer.handle,stream,dq.p,dk.p,dw.p,p0,T,ld,out.p,ids.p,
                                  scorer.key.p,scorer.key.n,scorer.dot.p,scorer.dot.n);};
        launch();ck(cudaStreamSynchronize(stream)); // initialize library before capture
        cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
        ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<2;++i)ck(cudaGraphLaunch(executable,stream));
        ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        blas(cublasSetStream(scorer.handle,nullptr));
        blas(cublasSetWorkspace(scorer.handle,scorer.workspace.p,Scorer::WS));ck(cudaStreamDestroy(stream));
    }
    const auto a=old.get(),b=out.get();const auto sel=ids.get(),prior=baseline.get();
    double sq=0,norm=0,max_abs=0,max_cpu=0,min_margin=INFINITY;size_t count=0,overlap=0,selected=0;
    std::vector<float> errors;errors.reserve(size_t(T)*visible);
    for(int t=0;t<T;++t){
        const int n=(p0+t+1)/4,top=std::min(512,n);
        std::vector<bool> seen(n),oldset(n);
        for(int i=0;i<top;++i){need(prior[t*512+i]>=0&&prior[t*512+i]<n,"F32 selection range");oldset[prior[t*512+i]]=true;}
        for(int i=0;i<top;++i){const int id=sel[t*512+i];need(id>=0&&id<n&&!seen[id],"candidate future/duplicate selection");
            seen[id]=true;overlap+=oldset[id];++selected;}
        for(int i=top;i<512;++i)need(sel[t*512+i]==-12345,"selection tail overwritten");
        double row_error=0;float next=-INFINITY;
        for(int j=0;j<n;++j){const size_t ix=size_t(t)*ld+j;need(std::isfinite(b[ix]),"nonfinite candidate score");
            double error=std::abs(double(b[ix])-a[ix]);max_abs=std::max(max_abs,error);row_error=std::max(row_error,error);
            sq+=error*error;norm+=double(a[ix])*a[ix];++count;errors.push_back(float(error));
            if(!oldset[j])next=std::max(next,a[ix]);}
        const float cut=a[size_t(t)*ld+prior[t*512+top-1]];
        if(n>top)min_margin=std::min(min_margin,double(cut)-next);
        for(int i=0;i<top;++i)need(double(a[size_t(t)*ld+sel[t*512+i]])+2*row_error>=cut,
            "selection disagreement outside measured F32 score envelope");
        for(int j=n;j<ld;++j)need(b[size_t(t)*ld+j]==12345.0f,"candidate wrote future score or row padding");
        if(t==0||t==T/2||t==T-1){
            // FP64 oracle samples spread across visible keys, plus the two selection boundaries.
            std::vector<int> probes;for(int i=0;i<16;++i)probes.push_back(i*(n-1)/15);
            probes.push_back(sel[t*512+top-1]);if(n>512)probes.push_back(prior[t*512+511]);
            for(int j:probes){double score=0;
                for(int h=0;h<HEADS;++h){double dot=0;for(int d=0;d<DIM;++d)
                    dot+=double(q[((size_t)t*HEADS+h)*DIM+d])*__half2float(__ushort_as_half(k[size_t(j)*DIM+d]));
                    score+=std::max(dot,0.0)*w[t*HEADS+h];}
                max_cpu=std::max(max_cpu,std::abs(double(b[size_t(t)*ld+j])-score));
            }
            std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](int x,int y){float sx=b[size_t(t)*ld+x],sy=b[size_t(t)*ld+y];return sx>sy||(sx==sy&&x<y);});
            for(int i=0;i<top;++i)need(sel[t*512+i]==order[i],"unchanged selector order/tie contract");
        }
    }
    for(size_t i=size_t(T)*ld;i<b.size();++i)need(b[i]==12345.0f,"score allocation guard");
    for(size_t i=size_t(T)*512;i<sel.size();++i)need(sel[i]==-12345,"selection allocation guard");
    const double rel=std::sqrt(sq/std::max(norm,1e-300));need(rel<1e-5,"cuBLAS scorer relative error");
    need(max_cpu<1e-4,"cuBLAS sampled FP64 error");
    char margin[64];if(std::isfinite(min_margin))std::snprintf(margin,sizeof(margin),"%.9g",min_margin);
    else std::snprintf(margin,sizeof(margin),"null");
    const size_t quantile=std::min(errors.size()-1,size_t(errors.size()*.99));
    std::nth_element(errors.begin(),errors.begin()+quantile,errors.end());
    std::printf("{\"case\":\"cublas_indexer\",\"T\":%d,\"p0\":%d,\"visible_max\":%d,\"mode\":%d,\"relative_l2_vs_f32\":%.9g,\"p99_abs_vs_f32\":%.9g,\"max_abs_vs_f32\":%.9g,\"sampled_max_abs_vs_fp64\":%.9g,\"minimum_f32_margin512_513\":%s,\"score_count\":%zu,\"selection_overlap\":%zu,\"selection_count\":%zu,\"caller_owned_workspace_bytes\":%zu,\"pass\":true}\n",
        T,p0,visible,mode,rel,errors[quantile],max_abs,max_cpu,margin,count,overlap,selected,Scorer::WS+(size_t(KK)*DIM+size_t(TT)*HEADS*KK)*sizeof(float));
    if(bench){for(int pair=0;pair<6;++pair){float f32,candidate;const int repeats=T>=256?1:3;
        if(pair&1){candidate=timing(newrun,repeats);f32=timing(oldrun,repeats);}else{f32=timing(oldrun,repeats);candidate=timing(newrun,repeats);}
        std::printf("{\"case\":\"cublas_stage_timing\",\"T\":%d,\"p0\":%d,\"pair\":%d,\"f32_ms\":%.9g,\"candidate_ms\":%.9g,\"repeats\":%d}\n",T,p0,pair,f32,candidate,repeats);}}
}
}
int main(int argc,char**){
    try{
        cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));need(p.major==7&&p.minor==0,"cuBLAS candidate requires actual SM70");
        Scorer scorer;int version=0;blas(cublasGetVersion(scorer.handle,&version));
        std::printf("{\"cublas_version\":%d,\"math\":\"FP32 SGEMM; no tensor-core claim\",\"query_tile\":%d,\"key_tile\":%d,\"epilogue_scores_per_warp\":%d,\"matrix_revision\":2,\"boundary_p0\":2052,\"selection\":\"T unchanged upstream single-token launches, included in timing\"}\n",version,TT,KK,ROWS_PER_WARP);
        bool bench=argc>1;
        if(bench){
            for(int T:{1,2,3,8,16,32,256})for(int p0:{2048,8192,32768,125000})run_case(scorer,T,p0,0,true);
            for(int p0:{65536,98304})run_case(scorer,256,p0,0,true);
            for(int T:{1024,2048,4096})for(int p0:{2048,8192,32768,65536,98304,120000})run_case(scorer,T,p0,0,true);
        }else{
            for(int T:{1,3,8,33})for(int p0:{2040,8192})run_case(scorer,T,p0,0,false);
            for(int mode:{1,2,3})run_case(scorer,3,2052,mode,false);
        }
        std::puts("{\"suite\":\"isolated_cublas_indexer\",\"pass\":true}");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"cuBLAS indexer FAILED: %s\n",e.what());return 1;}
}
#endif
