// Isolated P1 scorer -> unchanged upstream selection. No production dispatch.
#include "dsv4_volta_idx.cuh"
#include "strata/kernels/dsv4_fast.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace ds = strata::kernels::dsv4;
namespace v = strata::kernels::volta;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void need(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
template<class T> struct Buffer {
    T* p = nullptr; size_t n;
    explicit Buffer(const std::vector<T>& x): n(x.size()) {
        ck(cudaMalloc((void**)&p, n * sizeof(T)));
        ck(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<T> get() const {
        std::vector<T> x(n); ck(cudaMemcpy(x.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return x;
    }
};
uint32_t rng = 0x195714u;
float random_value() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int(rng % 65537) - 32768) / 32768.0f;
}
template<int W> void launch(const float* q, const uint16_t* k, const float* w, int n, float* s, cudaStream_t stream = nullptr) {
    v::idx_score_volta_kernel<W><<<(n + W*32-1)/(W*32), W*32, 0, stream>>>(q,k,w,n,s);
    ck(cudaGetLastError());
}
template<class F> float timing(F f) {
    for (int i=0;i<3;++i) f();
    cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a)); for (int i=0;i<20;++i) f(); ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
    float ms=0; ck(cudaEventElapsedTime(&ms,a,b)); ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    return ms * 1000 / 20;
}
template<int W> void run_case(int n, int mode, bool bench) {
    rng=0x195714u ^ uint32_t(n) ^ (uint32_t(mode)<<20); // Identical inputs for 4/8-warp candidates.
    std::vector<float> q(64*128), w(64), blank(n+32, 12345.0f);
    std::vector<uint16_t> k(size_t(n)*128);
    for (float& x:q) x=random_value();
    for (float& x:w) x=random_value(); // Includes negative head weights.
    for (auto& x:k) x=__half_as_ushort(__float2half_rn(random_value()));
    if (mode==1) std::fill(q.begin(), q.end(), 0.0f); // Exact ties.
    if (mode==2) for (auto& x:q) x *= 8192.0f;
    if (mode==3) for (auto& x:q) x *= std::ldexp(1.0f,-20);
    if (mode==4) { // Identical adjacent keys: controlled ties at the 512/513 boundary.
        for (int j=1;j<n;++j) if (j&1) std::copy_n(k.begin()+size_t(j-1)*128,128,k.begin()+size_t(j)*128);
    }
    for (float x:q) need(std::isfinite(x) && std::abs(x)<=65504, "query outside unscaled FP16 candidate domain");
    Buffer<float> dq(q), dw(w), out(blank), baseline(blank);
    Buffer<uint16_t> dk(k);
    Buffer<int> selected(std::vector<int>(512+32,-12345)), old_selected(std::vector<int>(512+32,-12345));
    launch<W>(dq.p,dk.p,dw.p,n,out.p);
    ds::indexer_scores(dq.p,dk.p,dw.p,64,n,baseline.p,nullptr);
    ds::indexer_select(out.p,n,512,selected.p,nullptr);
    ds::indexer_select(baseline.p,n,512,old_selected.p,nullptr);
    ck(cudaDeviceSynchronize());
    const auto got=out.get(), f32=baseline.get(); const auto ids=selected.get(), old=old_selected.get();
    std::vector<double> ref(n), errors(n); double sq=0, norm=0, max_abs=0, f32_max=0;
    for (int j=0;j<n;++j) {
        double score=0;
        for (int h=0;h<64;++h) {
            double dot=0;
            for (int d=0;d<128;++d) dot += double(q[h*128+d]) * __half2float(__ushort_as_half(k[size_t(j)*128+d]));
            score += std::max(dot,0.0) * w[h];
        }
        need(std::isfinite(got[j]), "nonfinite candidate score");
        ref[j]=score; errors[j]=std::abs(got[j]-score); max_abs=std::max(max_abs,errors[j]);
        f32_max=std::max(f32_max,std::abs(f32[j]-score)); sq+=errors[j]*errors[j]; norm+=score*score;
    }
    const double rel=std::sqrt(sq/std::max(norm,1e-300));
    // Small-query FP16 residual underflow is disclosed rather than silently clamped.
    need(mode==3 ? max_abs<2e-5 : rel<1e-4, "candidate FP64 scorer error exceeds investigation threshold");
    std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){return got[a]>got[b] || (got[a]==got[b] && a<b);});
    const int top=std::min(n,512); int overlap=0; std::vector<bool> seen(n), oldset(n);
    for(int i=0;i<top;++i) { need(old[i]>=0 && old[i]<n,"F32 selection range"); oldset[old[i]]=true; }
    auto sorted=ref; std::sort(sorted.begin(),sorted.end(),std::greater<double>());
    for(int i=0;i<top;++i) {
        need(ids[i]>=0 && ids[i]<n && !seen[ids[i]], "candidate selection range/duplicate");
        seen[ids[i]]=true; need(ids[i]==order[i],"unchanged selection order/tie contract");
        need(ref[ids[i]]+2*max_abs+1e-12>=sorted[top-1],"top-k disagreement beyond measured score envelope");
        overlap+=oldset[ids[i]];
    }
    for(int i=top;i<544;++i) need(ids[i]==-12345,"selection tail guard");
    for(int i=n;i<n+32;++i) need(got[i]==12345.0f,"score tail guard");
    // Graph replay uses the same whole-warp contract and starts accumulators afresh.
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); launch<W>(dq.p,dk.p,dw.p,n,out.p,stream);
    ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int i=0;i<3;++i) ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
    need(out.get()==got,"graph replay score repeatability");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    std::sort(errors.begin(),errors.end());
    cudaFuncAttributes attr{}; ck(cudaFuncGetAttributes(&attr,v::idx_score_volta_kernel<W>));
    int active=0; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,v::idx_score_volta_kernel<W>,W*32,0));
    std::printf("{\"case\":\"volta_indexer\",\"warps\":%d,\"visible\":%d,\"mode\":%d,\"relative_l2\":%.9g,\"p99_abs\":%.9g,\"max_abs\":%.9g,\"f32_max_abs\":%.9g,\"top512_overlap\":%d,\"margin512_513\":%.9g,\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"active_blocks_per_sm\":%d,\"pass\":true}\n",
        W,n,mode,rel,errors[std::min(n-1,int(n*.99))],max_abs,f32_max,overlap,n>512?sorted[511]-sorted[512]:0,attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,active);
    if(bench) {
        for(int pair=0;pair<6;++pair) {
            float a,b;
            auto oldrun=[&]{ds::indexer_scores(dq.p,dk.p,dw.p,64,n,baseline.p,nullptr);};
            auto newrun=[&]{launch<W>(dq.p,dk.p,dw.p,n,out.p);};
            if(pair&1) {b=timing(newrun);a=timing(oldrun);} else {a=timing(oldrun);b=timing(newrun);}
            const float select=timing([&]{ds::indexer_select(out.p,n,512,selected.p,nullptr);});
            std::printf("{\"case\":\"isolated_timing\",\"warps\":%d,\"visible\":%d,\"pair\":%d,\"f32_us\":%.6f,\"volta_us\":%.6f,\"select_us\":%.6f}\n",W,n,pair,a,b,select);
        }
    }
}
} // namespace
int main(int argc,char**) {
    try {
        cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0)); need(p.major==7 && p.minor==0,"isolated Volta scorer requires SM70");
        const bool bench=argc>1;
        for(int n:{1,7,31,129,512,513,1024,8192,31250}) {run_case<4>(n,0,bench);run_case<8>(n,0,bench);}
        for(int mode:{1,2,3,4}) {run_case<4>(513,mode,false);run_case<8>(513,mode,false);}
        std::puts("{\"suite\":\"isolated_volta_indexer\",\"pass\":true}"); return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Volta indexer FAILED: %s\n",e.what());return 1;}
}
