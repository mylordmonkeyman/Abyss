// Isolated original/batched/sequential/adaptive/cutoff cuBLAS comparison.
#ifndef STRATA_USE_HIP
#include "../../src/kernels/cuda/dsv4_cublas_corrected.cuh"
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
const char* labels[]={"original","batched-warp","batched-sequential","full-flagged","band-flagged","band-all","band-all-sequential"};
struct Scorer {
    int variant;uint32_t* work=nullptr;int* state=nullptr;size_t work_count=0,state_count=0;
    // Scratch is capped independently of the common caller-owned score/selection buffers.
    static constexpr size_t WS=16*1024*1024;
    static_assert(WS+(size_t(KK)*DIM+size_t(TT)*HEADS*KK)*sizeof(float)<64*1024*1024,
        "caller-owned cuBLAS scratch cap");
    Buffer<float> key{size_t(KK)*DIM},dot{size_t(TT)*HEADS*KK};Buffer<unsigned char> workspace{WS};
    cublasHandle_t handle{};
    explicit Scorer(int v):variant(v){blas(cublasCreate(&handle));blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
        blas(cublasSetWorkspace(handle,workspace.p,WS));}
    ~Scorer(){cublasDestroy(handle);}
    void run(const float* q,const uint16_t* k,const float* w,int p0,int T,int ld,float* score,int* ids,cudaStream_t stream=nullptr){
        if(variant==0){ds::p1::run(handle,stream,q,k,w,p0,T,ld,score,ids,key.p,key.n,dot.p,dot.n);return;}
        using Policy=ds::p1::correction::Policy;
        const auto policy=variant<=2?Policy::none:variant==3?Policy::full_flagged:variant==4?Policy::band_flagged:Policy::band_all;
        ds::p1::correction::run(handle,stream,q,k,w,p0,T,ld,score,ids,key.p,key.n,dot.p,dot.n,
            work,work_count,state,state_count,policy,variant==2||variant==6);
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
    if(mode==4)for(int j=1;j<visible;++j){
        std::copy_n(k.begin(),DIM,k.begin()+size_t(j)*DIM);
        const float first=__half2float(__ushort_as_half(k[size_t(j)*DIM]));
        k[size_t(j)*DIM]=__half_as_ushort(__float2half_rn(first+(j%3-1)*0.00048828125f));
    }
    Buffer<uint32_t> work(scorer.variant?size_t(T)*ds::p1::correction::Capacity:1);
    Buffer<int> state(scorer.variant?size_t(T)*ds::p1::correction::StateWords:1);
    scorer.work=work.p;scorer.work_count=work.n;scorer.state=state.p;scorer.state_count=state.n;
    Buffer<float> dq(q),dw(w),out(blank),old(blank);Buffer<uint16_t> dk(k);
    Buffer<int> ids(std::vector<int>(size_t(T)*512+32,-12345)),baseline(std::vector<int>(size_t(T)*512+32,-12345));
    auto oldrun=[&]{ds::indexer_batch(dq.p,dk.p,dw.p,HEADS,4,p0,T,512,old.p,ld,baseline.p,nullptr);};
    auto newrun=[&]{scorer.run(dq.p,dk.p,dw.p,p0,T,ld,out.p,ids.p);};
    oldrun();newrun();ck(cudaDeviceSynchronize());
    if(!bench){
        const auto before=out.get();
        cudaStream_t stream;ck(cudaStreamCreate(&stream));
        // Set stream before workspace: cuBLAS resets workspace when stream changes.
        blas(cublasSetStream(scorer.handle,stream));
        blas(cublasSetWorkspace(scorer.handle,scorer.workspace.p,Scorer::WS));
        auto launch=[&]{scorer.run(dq.p,dk.p,dw.p,p0,T,ld,out.p,ids.p,stream);};
        launch();ck(cudaStreamSynchronize(stream)); // initialize library before capture
        cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
        ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<2;++i)ck(cudaGraphLaunch(executable,stream));
        ck(cudaStreamSynchronize(stream));need(out.get()==before,"graph score repeatability");ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        blas(cublasSetStream(scorer.handle,nullptr));
        blas(cublasSetWorkspace(scorer.handle,scorer.workspace.p,Scorer::WS));ck(cudaStreamDestroy(stream));
    }
    const auto a=old.get(),b=out.get();const auto sel=ids.get(),prior=baseline.get();
    const auto counters=state.get();size_t flagged=0,corrected=0,rescored=0,unflagged_order_errors=0,unflagged_membership_errors=0;
    for(int t=0;t<T;++t){
        const int top=std::min(512,(p0+t+1)/4),count=scorer.variant?counters[t]:0;
        const bool flag=scorer.variant&&counters[T+t]!=0;flagged+=flag;corrected+=count>0;rescored+=count;
        need(count>=0&&count<=(p0+t+1)/4,"invalid GPU correction count");
        bool same=true;
        for(int i=0;i<top;++i)same&=sel[t*512+i]==prior[t*512+i];
        bool membership=true;
        if(!count){
            if(!same)++unflagged_order_errors;
            for(int i=0;i<top;++i)membership&=std::find(prior.begin()+t*512,prior.begin()+t*512+top,sel[t*512+i])!=prior.begin()+t*512+top;
            if(!membership)++unflagged_membership_errors;
        }
        if(scorer.variant>=3){
            if(count){need(same,"corrected selected order differs from operational F32");
                for(int i=0;i<top;++i){const size_t index=size_t(t)*ld+sel[t*512+i];need(b[index]==a[index],"corrected selected score differs from operational F32");}}
            else{
                need(membership,"unflagged membership error");
            }
        }
    }
    std::printf("{\"case\":\"correction\",\"variant\":\"%s\",\"T\":%d,\"p0\":%d,\"mode\":%d,\"flagged_tokens\":%zu,\"corrected_tokens\":%zu,\"rescored_rows\":%zu,\"unflagged_order_errors\":%zu,\"exact_corrected_scores_and_order\":%s,\"unflagged_membership_errors\":%zu,\"pass\":true}\n",labels[scorer.variant],T,p0,mode,flagged,corrected,rescored,unflagged_order_errors,scorer.variant>=3?"true":"false",unflagged_membership_errors);
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
    std::printf("{\"case\":\"cublas_indexer\",\"variant\":\"%s\",\"T\":%d,\"p0\":%d,\"visible_max\":%d,\"mode\":%d,\"relative_l2_vs_f32\":%.9g,\"p99_abs_vs_f32\":%.9g,\"max_abs_vs_f32\":%.9g,\"sampled_max_abs_vs_fp64\":%.9g,\"minimum_f32_margin512_513\":%s,\"score_count\":%zu,\"selection_overlap\":%zu,\"selection_count\":%zu,\"caller_owned_workspace_bytes\":%zu,\"pass\":true}\n",
        labels[scorer.variant],T,p0,visible,mode,rel,errors[quantile],max_abs,max_cpu,margin,count,overlap,selected,Scorer::WS+(size_t(KK)*DIM+size_t(TT)*HEADS*KK+work.n+state.n)*sizeof(float));
    if(bench){for(int pair=0;pair<6;++pair){float f32,candidate;const int repeats=T>=256?1:3;
        if(pair&1){candidate=timing(newrun,repeats);f32=timing(oldrun,repeats);}else{f32=timing(oldrun,repeats);candidate=timing(newrun,repeats);}
        std::printf("{\"case\":\"cublas_stage_timing\",\"variant\":\"%s\",\"T\":%d,\"p0\":%d,\"pair\":%d,\"f32_ms\":%.9g,\"candidate_ms\":%.9g,\"repeats\":%d}\n",labels[scorer.variant],T,p0,pair,f32,candidate,repeats);}}
}
}
int main(int argc,char**){
    try{
        cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));need(p.major==7&&p.minor==0,"cuBLAS candidate requires actual SM70");
        std::puts("{\"matrix_revision\":1,\"scope\":\"isolated cuBLAS correction variants, empirical 32eps envelope, unchanged F32 reference\"}");
        const bool bench=argc>1;
        for(int variant=0;variant<7;++variant){
            Scorer scorer(variant);
            auto resource=[&](const char* kernel,auto function,int threads){
                cudaFuncAttributes a{};int active=0;ck(cudaFuncGetAttributes(&a,function));
                ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,threads,0));
                std::printf("{\"case\":\"resources\",\"variant\":\"%s\",\"kernel\":\"%s\",\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"active_blocks_per_sm\":%d}\n",labels[variant],kernel,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,active);
            };
            namespace c=ds::p1::correction;
            if(variant==2||variant==6)resource("sequential-epilogue",c::sequential_epilogue,256);
            if(variant>=3){
                resource("detector",c::detect,256);
                if(variant==3)resource("rescore",c::rescore<true,false>,256);
                else if(variant==4)resource("rescore",c::rescore<false,false>,256);
                else resource("rescore",c::rescore<false,true>,256);
            }
            if(bench){
                for(int T:{256,1024,4096})for(int p0:{2048,8192,32768,120000})run_case(scorer,T,p0,0,true);
                for(int p0:{65536,98304,120000})run_case(scorer,2048,p0,0,true);
            }else{
                for(int T:{1,3,8,9,33})for(int p0:{2040,2052,8192})run_case(scorer,T,p0,0,false);
                for(int mode:{1,2,3,4})run_case(scorer,3,2052,mode,false);
                run_case(scorer,3,40000,1,false);
            }
        }
        std::puts("{\"suite\":\"isolated_cublas_corrections\",\"pass\":true}");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"cuBLAS indexer FAILED: %s\n",e.what());return 1;}
}
#endif
