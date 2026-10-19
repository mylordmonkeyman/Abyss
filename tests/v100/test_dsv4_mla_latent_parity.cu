// P2a addressing + staged arithmetic. Synthetic cached rows, actual Abyss
// win_load/win_store/q_prep_batch/attend_batch. Not a fused production path.
#ifndef STRATA_USE_HIP
#include "qkpv_math.cuh"
#include "strata/kernels/dsv4_fast.hpp"
#include <numeric>
#include <filesystem>
#include <fstream>
#include <cublas_v2.h>

using namespace v100diag;
namespace ds = strata::kernels::dsv4;
namespace {
constexpr int WINDOW=128,TOP=512;
float hf(uint16_t v) { return __half2float(__ushort_as_half(v)); }
uint16_t fh(float v) { return __half_as_ushort(__float2half_rn(v)); }

// Checked diagnostic gather: invalid IDs never become addresses. Future and
// duplicate selections are errors, not silently accepted softmax entries.
__global__ void gather(const uint16_t* raw,const uint16_t* comp,const int* sel,
                       int hist,int p0,int t,int ratio,int comp_rows,
                       uint16_t* dst,int* ids,int* status) {
    const int j=blockIdx.x,d=threadIdx.x,p=p0+t,nr=min(p+1,WINDOW);
    const int visible=ratio?(p+1)/ratio:0;
    const int nc=ratio==4?min(TOP,visible):visible;
    if(j>=nr+nc)return;
    int row=0;bool bad=false;
    if(j<nr) {
        row=p-nr+1-(p0-hist)+j;
        bad=row<0 || row>=hist+t+1;
        if(d==0)ids[j]=p-nr+1+j;
    } else {
        const int k=j-nr;row=ratio==4?sel[k]:k;
        bad=row<0 || row>=visible || row>=comp_rows;
        if(ratio==4)for(int i=0;i<k;++i)bad|=sel[i]==row;
        if(d==0)ids[j]=-row-1;
    }
    if(bad) { if(d==0)atomicOr(status,1);dst[j*D+d]=0;return; }
    dst[j*D+d]=(j<nr?raw:comp)[row*D+d];
}

__global__ void inverse_rope(const float* input,float* output,ds::Rope r,int pos) {
    const int h=blockIdx.x,d=threadIdx.x;
    if(d<D-64) {output[h*D+d]=input[h*D+d];return;}
    const int i=d-(D-64);if(i&1)return;
    const float extrap=float(pos)*powf(r.theta_scale,i/2.0f),interp=r.freq_scale*extrap;
    float angle=interp,mag=r.attn_factor;
    if(r.ext_factor!=0) {
        const float y=(float(i/2)-r.corr_lo)/fmaxf(.001f,r.corr_hi-r.corr_lo);
        const float mix=(1-fminf(1.f,fmaxf(0.f,y)))*r.ext_factor;
        angle=interp*(1-mix)+extrap*mix;
        mag*=1+.1f*logf(1/r.freq_scale);
    }
    const float c=cosf(angle)*mag,s=-sinf(angle)*mag,a=input[h*D+d],b=input[h*D+d+1];
    output[h*D+d]=a*c-b*s;output[h*D+d+1]=a*s+b*c;
}

void rotate_host(std::vector<double>& x,const ds::Rope& r,int pos,bool forward) {
    for(int i=0;i<64;i+=2) {
        const float ex=float(pos)*std::pow(r.theta_scale,i/2.0f),in=r.freq_scale*ex;
        float theta=in,mag=r.attn_factor;
        if(r.ext_factor!=0) {
            const float y=(float(i/2)-r.corr_lo)/std::max(.001f,r.corr_hi-r.corr_lo);
            const float mix=(1-std::clamp(y,0.f,1.f))*r.ext_factor;
            theta=in*(1-mix)+ex*mix;mag*=1+.1f*std::log(1/r.freq_scale);
        }
        const double c=std::cos(double(theta))*mag,s=std::sin(double(theta))*mag*(forward?1:-1);
        const int d=D-64+i;const double a=x[d],b=x[d+1];
        x[d]=a*c-b*s;x[d+1]=a*s+b*c;
    }
}

#include "latent_components.cuh"

void run_case(int ratio,int p0,int T,bool yarn,OutputProjection& projection) {
    const int hist=std::min(p0,127),rows=hist+T,comp_rows=ratio?(p0+T)/ratio:0;
    const auto rope=ds::make_rope(64,yarn?160000.f:10000.f,yarn?1.f/16:1.f,
                                 yarn?1.f:0.f,4096,32,1);
    std::mt19937 rng(91217+ratio+p0+T);std::uniform_real_distribution<float> uniform(-.5f,.5f);
    std::vector<float> query(T*H*D),sinks(H);
    for(auto& v:query)v=uniform(rng);
    for(auto& v:sinks)v=uniform(rng);
    sinks[0]=20;sinks[1]=-20;
    std::vector<uint16_t> keys(rows*D),comp((comp_rows+1)*D,fh(NAN)),ring(WINDOW*D,fh(NAN));
    // Cache rows already contain RoPE, at each raw position or compressor BLOCK
    // START. This does not test learned projections, norms or compressor creation.
    auto cache_row=[&](uint16_t* dst,int pos) {
        std::vector<double> x(D);for(auto& v:x)v=uniform(rng);
        rotate_host(x,rope,pos,true);for(int d=0;d<D;++d)dst[d]=fh(float(x[d]));
    };
    for(int r=0;r<rows;++r)cache_row(keys.data()+r*D,p0-hist+r);
    if(ratio)comp=compressed_rows(ratio,p0,T,rope);
    for(int r=0;r<hist;++r)std::copy_n(keys.data()+r*D,D,ring.data()+((p0-hist+r)%WINDOW)*D);
    std::vector<int> selections(T*TOP,-12345);
    for(int t=0;t<T && ratio==4;++t) {
        const int visible=(p0+t+1)/4;
        for(int j=0;j<std::min(TOP,visible);++j)
            selections[t*TOP+j]=(j&1)?j/2:visible-1-j/2;
    }
    Buffer<float> dq(query),dsink(sinks),fallback(std::vector<float>(T*H*D+GUARD,sentinel));
    Buffer<uint16_t> dr(ring),dk(keys),dc(comp);
    Buffer<int> dsel(selections);
    // The real ring-to-linear API must reproduce every historical row exactly.
    ds::win_load(dr.p,WINDOW,dk.p,hist,p0,D,nullptr);ck(cudaDeviceSynchronize());
    need(dk.get()==keys,"win_load cached-row identity");
    ds::q_prep_batch(dq.p,H,D,64,rope,nullptr,0,0,rope,p0,T,1e-6f,nullptr);
    ds::attend_batch(dq.p,dk.p,hist,WINDOW,dc.p,dsel.p,TOP,ratio,dsink.p,H,D,64,rope,p0,T,fallback.p,nullptr);
    ck(cudaDeviceSynchronize());const auto prepared=dq.get(),f32=fallback.get();
    for(size_t i=T*H*D;i<f32.size();++i)need(f32[i]==sentinel,"fallback guard");
    // Reconstruct the exact same keys through a split prefill's real store/load.
    const int split=std::max(1,T/2),second=T-split;
    Buffer<uint16_t> split_ring(ring);
    ds::win_store(split_ring.p,WINDOW,dk.p,hist,p0,split,D,nullptr);
    const int p1=p0+split,hist1=std::min(p1,127);
    std::vector<uint16_t> keys1((hist1+second)*D,fh(NAN));
    if(second)std::copy_n(keys.data()+(hist+split)*D,second*D,keys1.data()+hist1*D);
    Buffer<uint16_t> dk1(keys1);
    ds::win_load(split_ring.p,WINDOW,dk1.p,hist1,p1,D,nullptr);ck(cudaDeviceSynchronize());
    const auto loaded1=dk1.get();
    const int start=p1-hist1-(p0-hist);
    need(std::equal(loaded1.begin(),loaded1.end(),keys.begin()+start*D),"split window row identity");
    if(second) {
        Buffer<float> split_out(std::vector<float>(second*H*D+GUARD,sentinel));
        ds::attend_batch(dq.p+split*H*D,dk1.p,hist1,WINDOW,dc.p,dsel.p+split*TOP,TOP,ratio,
                         dsink.p,H,D,64,rope,p1,second,split_out.p,nullptr);
        ck(cudaDeviceSynchronize());const auto got=split_out.get();
        for(int i=0;i<second*H*D;++i)need(got[i]==f32[split*H*D+i],"fallback split equality");
        for(size_t i=second*H*D;i<got.size();++i)need(got[i]==sentinel,"split output guard");
    }
    std::vector<int> tokens={0,T-1};
    for(int t=1;t<T;++t)if((p0+t+1)%128==0 || (ratio==4 && (p0+t+1)%4==0))tokens.push_back(t);
    std::sort(tokens.begin(),tokens.end());tokens.erase(std::unique(tokens.begin(),tokens.end()),tokens.end());
    double worst=0,worst_f32=0,worst_qk=0,worst_pv=0;
    for(int t:tokens) {
        const int p=p0+t,nr=std::min(p+1,WINDOW),vis=ratio?(p+1)/ratio:0,nc=ratio==4?std::min(TOP,vis):vis,n=nr+nc;
        Buffer<uint16_t> gathered(std::vector<uint16_t>(n*D+GUARD,0x7bff));
        Buffer<int> ids(std::vector<int>(n+GUARD,-12345)),status(std::vector<int>(1,0));
        gather<<<n,D>>>(dk.p,dc.p,dsel.p+t*TOP,hist,p0,t,ratio,comp_rows,gathered.p,ids.p,status.p);
        ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(status.get()[0]==0,"valid gather rejected");
        const auto kv=gathered.get();const auto row_ids=ids.get();std::vector<int> expected_ids;
        for(int j=0;j<n;++j) {
            const int r=j<nr?p-nr+1+j-(p0-hist):ratio==4?selections[t*TOP+j-nr]:j-nr;
            expected_ids.push_back(j<nr?p-nr+1+j:-r-1);
            const auto* src=(j<nr?keys.data():comp.data())+r*D;
            need(std::equal(src,src+D,kv.begin()+j*D),"gather key bytes/order");
            need(row_ids[j]==expected_ids.back(),"gather absolute row identity");
        }
        for(size_t i=n*D;i<kv.size();++i)need(kv[i]==0x7bff,"gather key guard");
        for(size_t i=n;i<row_ids.size();++i)need(row_ids[i]==-12345,"gather ID guard");
        Buffer<float> scores(std::vector<float>(H*n+GUARD,sentinel)),prob(std::vector<float>(H*n+GUARD,sentinel)),
                      den(std::vector<float>(H+GUARD,sentinel)),pv(std::vector<float>(H*D+GUARD,sentinel)),
                      out(std::vector<float>(H*D+GUARD,sentinel));
        auto launch=[&](cudaStream_t stream) {
            qk_mma<<<dim3(H/8,(n+31)/32),32,0,stream>>>(dq.p+t*H*D,gathered.p,scores.p,n);
            softmax_sink<<<1,H,0,stream>>>(scores.p,dsink.p,prob.p,den.p,n);
            pv_mma<<<dim3(H/8,D/32),32,0,stream>>>(prob.p,gathered.p,pv.p,n,1024);
            inverse_rope<<<H,D,0,stream>>>(pv.p,out.p,rope,p);
        };
        launch(nullptr);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
        cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(stream);
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<3;++i)ck(cudaGraphLaunch(executable,stream));
        ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
        // Independent FP64 equations use HOST row identities, never gathered
        // device rows, so a matching gather bug cannot conceal a semantic miss.
        std::vector<double> qk(H*n),pr(H*n),dn(H),latent(H*D),rotated(H*D),existing(H*D);
        for(int h=0;h<H;++h) {
            double mx=sinks[h];
            for(int j=0;j<n;++j) {
                const int id=expected_ids[j];const auto* src=id>=0?keys.data()+(id-(p0-hist))*D:comp.data()+(-id-1)*D;
                double dot=0;for(int d=0;d<D;++d)dot+=double(prepared[(t*H+h)*D+d])*hf(src[d]);
                qk[h*n+j]=dot;mx=std::max(mx,dot/std::sqrt(double(D)));
            }
            dn[h]=std::exp(sinks[h]-mx);
            for(int j=0;j<n;++j)dn[h]+=std::exp(qk[h*n+j]/std::sqrt(double(D))-mx);
            std::vector<double> z(D);
            for(int j=0;j<n;++j) {
                pr[h*n+j]=std::exp(qk[h*n+j]/std::sqrt(double(D))-mx)/dn[h];
                const int id=expected_ids[j];const auto* src=id>=0?keys.data()+(id-(p0-hist))*D:comp.data()+(-id-1)*D;
                for(int d=0;d<D;++d)z[d]+=pr[h*n+j]*hf(src[d]);
            }
            std::copy(z.begin(),z.end(),latent.begin()+h*D);rotate_host(z,rope,p,false);
            std::copy(z.begin(),z.end(),rotated.begin()+h*D);
            for(int d=0;d<D;++d)existing[h*D+d]=f32[(t*H+h)*D+d];
        }
        const auto eq=compare(scores.get(),qk),ep=compare(prob.get(),pr),ed=compare(den.get(),dn),
                   ev=compare(pv.get(),latent),eo=compare(out.get(),rotated),ef=compare(out.get(),existing);
        need(eq.rel<1e-5 && ep.rel<1e-5 && ed.rel<1e-5,"gathered QK/softmax screen");
        need(ev.rel<1e-5 && eo.rel<1e-5 && ef.rel<1e-5,"gathered PV/inverse RoPE screen");
        projection.check(out.p,fallback.p+t*H*D,out.get(),existing,ratio,p0,t);
        auto stage=[&](const char* name,const Error& e) {
            std::printf("{\"stage\":\"%s\",\"ratio\":%d,\"p0\":%d,\"t\":%d,"
                        "\"relative_l2\":%.9g,\"max_abs\":%.9g,\"p99_abs\":%.9g,\"pass\":true}\n",
                        name,ratio,p0,t,e.rel,e.max_abs,e.p99_abs);
        };
        stage("qk",eq);stage("probability",ep);stage("sink_denominator",ed);
        stage("pv",ev);stage("inverse_rope",eo);stage("versus_existing",ef);
        worst=std::max(worst,eo.rel);worst_f32=std::max(worst_f32,ef.rel);
        worst_qk=std::max(worst_qk,eq.rel);worst_pv=std::max(worst_pv,ev.rel);
        // Before attempting any bad address, reject negative, future and
        // duplicate CSA rows. Never pass malformed input into the upstream API.
        if(ratio==4 && nc>1)for(int mode=0;mode<3;++mode) {
            auto bad=std::vector<int>(selections.begin()+t*TOP,selections.begin()+(t+1)*TOP);
            bad[0]=mode==0?-1:mode==1?vis:bad[1];Buffer<int> dbad(bad),error(std::vector<int>(1,0));
            gather<<<n,D>>>(dk.p,dc.p,dbad.p,hist,p0,t,ratio,comp_rows,gathered.p,ids.p,error.p);
            ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(error.get()[0]==1,"malformed selection accepted");
        }
    }
    std::printf("{\"case\":\"cached_latent_gathers\",\"ratio\":%d,\"p0\":%d,\"T\":%d,\"yarn\":%s,"
                "\"checked_tokens\":%zu,\"qk_relative_l2_max\":%.9g,\"pv_relative_l2_max\":%.9g,"
                "\"inverse_rope_relative_l2_max\":%.9g,\"versus_existing_relative_l2_max\":%.9g,"
                "\"split_identity\":true,\"production_enabled\":false,\"pass\":true}\n",
                ratio,p0,T,yarn?"true":"false",tokens.size(),worst_qk,worst_pv,worst,worst_f32);
}
#include "real_stage_replay.cuh"
}
int main(int argc,char** argv) {
    try {
        cudaDeviceProp device{};ck(cudaGetDeviceProperties(&device,0));
        need(device.major==7 && device.minor==0,"latent diagnostic requires actual SM70");
        if(argc==2) {real_stages(argv[1]);return 0;}
        need(argc==1,"expected optional real-stage capture directory");
        OutputProjection projection;
        for(int T:{1,7,8,15,16,31,32,33,127,128,129})run_case(0,0,T,false,projection);
        run_case(0,127,7,false,projection);run_case(0,129,8,false,projection);
        for(int ratio:{4,128}) {
            run_case(ratio,ratio-2,7,true,projection);run_case(ratio,2046,7,true,projection);
            run_case(ratio,4094,7,true,projection);run_case(ratio,8190,7,true,projection);
            const auto rope=ds::make_rope(64,160000.f,1.f/16,1.f,4096,32,1);
            compressed_rows(ratio,ratio-2,1,rope);
        }
        std::puts("{\"suite\":\"cached_latent_gathers\",\"cases\":21,\"projection_compressor_gate_complete\":false,\"pass\":true}");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"latent diagnostic FAILED: %s\n",e.what());return 1;}
}
#endif
