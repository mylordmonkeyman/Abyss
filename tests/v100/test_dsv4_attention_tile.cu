// Isolated P2 tile feasibility. Existing attend_batch is the FP32 comparator;
// A valid 64-D identity RoPE isolates latent attention. Full model RoPE/projection
// gates remain separate; n_rot=0 is invalid in the upstream comparator.
#ifndef STRATA_USE_HIP
#include "attention_tile.cuh"
#include "strata/kernels/dsv4_fast.hpp"
#include <string>
namespace ds=strata::kernels::dsv4;
using namespace v100diag;

template<class F> float elapsed(F run){
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a));for(int i=0;i<2;++i)run();ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));
    float ms=0;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/2;
}

template<int K,int W> void test(int ratio,int p0,int T,int mode,bool bench){
    constexpr int shared=AttentionTile<K,W>::SHARED;
    ck(cudaFuncSetAttribute(attention_tile<K,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes attr{};ck(cudaFuncGetAttributes(&attr,attention_tile<K,W>));
    int occupancy=0;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,attention_tile<K,W>,W*32,shared));
    std::printf("{\"resources\":\"attention_tile\",\"key_tile\":%d,\"warps\":%d,\"shared_bytes\":%d,\"registers\":%d,\"local_bytes\":%zu,\"active_ctas_per_sm\":%d}\n",
                K,W,shared,attr.numRegs,attr.localSizeBytes,occupancy);
    need(attr.localSizeBytes==0 && occupancy>0,"unexpected attention tile spill/resource failure");
    const int hist=std::min(std::max(p0,0),127),comp_rows=ratio?(p0+T)/ratio:0;
    std::mt19937 rng(19937+ratio+p0+T+mode);std::uniform_real_distribution<float> random(-1.f,1.f);
    std::vector<float> query((size_t)T*H*D),sinks(H);
    std::vector<uint16_t> raw((size_t)(hist+T)*D),comp(std::max(size_t(1),(size_t)comp_rows*D));
    std::vector<int> ids((size_t)T*512,0);
    for(auto& x:query)x=mode==1?0.f:random(rng)*(mode==2?std::ldexp(1.f,-20):mode==3?32.f:1.f);
    for(auto& x:sinks)x=mode==3?12.f:random(rng)*3;
    for(auto& x:raw)x=__half_as_ushort(__float2half_rn(random(rng)));
    for(auto& x:comp)x=__half_as_ushort(__float2half_rn(random(rng)));
    for(int t=0;t<T;++t){
        const int visible=ratio?(p0+t+1)/ratio:0;
        for(int i=0;i<std::min(512,visible);++i)ids[(size_t)t*512+i]=visible-1-i;
    }
    const size_t count=(size_t)T*H*D;
    Buffer<float> dq(query),sink(sinks),out(std::vector<float>(count+GUARD,sentinel)),old(std::vector<float>(count+GUARD,sentinel));
    Buffer<uint16_t> dr(raw),dc(comp);Buffer<int> sel(ids);
    auto candidate=[&](cudaStream_t stream){
        attention_tile<K,W><<<dim3(T,H/8),W*32,shared,stream>>>(dq.p,dr.p,dc.p,sel.p,sink.p,hist,p0,ratio,T,out.p);
        ck(cudaGetLastError());
    };
    ds::Rope identity_rope{};identity_rope.freq_scale=0.f;
    auto baseline=[&]{ds::attend_batch(dq.p,dr.p,hist,128,dc.p,sel.p,512,ratio,sink.p,H,D,64,identity_rope,p0,T,old.p,nullptr);};
    candidate(nullptr);baseline();ck(cudaDeviceSynchronize());
    if(!bench){
        cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));candidate(stream);
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<2;++i)ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
        ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    }
    const auto got=out.get(),ref=old.get();
    double squared=0,ref_squared=0,maximum=0;
    for(size_t i=0;i<count;++i){need(std::isfinite(got[i]) && std::isfinite(ref[i]),"nonfinite attention");
        const double d=double(got[i])-ref[i];squared+=d*d;ref_squared+=double(ref[i])*ref[i];maximum=std::max(maximum,std::abs(d));}
    const double relative=std::sqrt(squared/std::max(ref_squared,1e-300));
    for(size_t i=count;i<got.size();++i)need(got[i]==sentinel && ref[i]==sentinel,"attention output guard");
    // Independent FP64 first-query/head oracle checks the actual gather and
    // denominator-only sink, not merely candidate agreement with another kernel.
    const int nr=std::min(p0+1,128),visible=ratio?(p0+1)/ratio:0,nc=ratio==4?std::min(512,visible):visible;
    std::vector<double> logits(nr+nc),expected(D,0);double m=sinks[0];
    auto key=[&](int j,int d){const auto v=j<nr?raw[(size_t)(p0-nr+1-(p0-hist)+j)*D+d]:
        comp[(size_t)(ratio==4?ids[j-nr]:j-nr)*D+d];return double(__half2float(__ushort_as_half(v)));};
    for(int j=0;j<nr+nc;++j){double value=0;for(int d=0;d<D;++d)value+=double(query[d])*key(j,d);
        logits[j]=value/std::sqrt(double(D));m=std::max(m,logits[j]);}
    double den=std::exp(double(sinks[0])-m);
    for(int j=0;j<nr+nc;++j){const double weight=std::exp(logits[j]-m);den+=weight;
        for(int d=0;d<D;++d)expected[d]+=weight*key(j,d);}
    double error=0,norm=0;for(int d=0;d<D;++d){expected[d]/=den;const double e=got[d]-expected[d];error+=e*e;norm+=expected[d]*expected[d];}
    const double oracle=std::sqrt(error/std::max(norm,1e-300));
    const bool passed=relative<1e-5 && maximum<.001 && oracle<1e-5;
    std::printf("{\"case\":\"attention_tile\",\"ratio\":%d,\"p0\":%d,\"T\":%d,\"mode\":%d,\"key_tile\":%d,\"warps\":%d,"
        "\"relative_l2\":%.9g,\"max_abs\":%.9g,\"oracle_relative_l2\":%.9g,\"shared_bytes\":%d,\"registers\":%d,"
        "\"local_bytes\":%zu,\"active_ctas_per_sm\":%d,\"production_enabled\":false,\"pass\":%s}\n",
        ratio,p0,T,mode,K,W,relative,maximum,oracle,shared,attr.numRegs,attr.localSizeBytes,occupancy,passed?"true":"false");
    need(passed,"tile versus valid-identity-RoPE FP32 / FP64 screen");
    if(bench)for(int pair=0;pair<6;++pair){
        float a,b;if(pair&1){b=elapsed([&]{candidate(nullptr);});a=elapsed(baseline);}
        else {a=elapsed(baseline);b=elapsed([&]{candidate(nullptr);});}
        std::printf("{\"timing\":\"attention_tile\",\"ratio\":%d,\"p0\":%d,\"T\":%d,\"key_tile\":%d,\"warps\":%d,\"pair\":%d,"
            "\"f32_ms\":%.9g,\"candidate_ms\":%.9g,\"scope\":\"synthetic prefill latent attention, valid identity RoPE; no model speed claim\"}\n",ratio,p0,T,K,W,pair,a,b);
    }
}
template<int K,int W> void suite(bool bench){
    if(bench){for(int T:{256,2048}){test<K,W>(0,131071,T,0,true);test<K,W>(4,131071,T,0,true);test<K,W>(128,131071,T,0,true);}}
    else {test<K,W>(0,0,1,0,false);test<K,W>(0,0,33,1,false);test<K,W>(0,127,7,2,false);
        test<K,W>(0,129,8,3,false);test<K,W>(4,2,7,0,false);test<K,W>(4,2047,8,0,false);
        test<K,W>(128,126,7,0,false);test<K,W>(128,131071,1,0,false);}
}
int main(int argc,char** argv){try{
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));need(prop.major==7 && prop.minor==0,"tile test needs physical SM70");
    const bool bench=argc==2 && std::string(argv[1])=="--bench";need(argc==1 || bench,"expected optional --bench");
    suite<16,4>(bench);suite<16,8>(bench);suite<32,4>(bench);suite<32,8>(bench);
    std::puts("{\"suite\":\"attention_tile\",\"production_enabled\":false,\"pass\":true}");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"P2 attention tile FAILED: %s\n",e.what());return 1;}}
#endif
