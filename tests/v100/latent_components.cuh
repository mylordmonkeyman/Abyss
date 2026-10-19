// Included by the CUDA-only latent diagnostic after its host RoPE oracle.
// Synthetic projected inputs and sparse F16 weights; no learned-weight claim.
#pragma once
#ifndef STRATA_USE_HIP

std::vector<uint16_t> compressed_rows(int ratio,int p0,int T,const ds::Rope& rope) {
    const int overlap=ratio==4,W=(overlap?2:1)*D,state=(overlap?2:1)*ratio,N=p0+T,nc=N/ratio;
    std::mt19937 rng(1783+N+ratio);std::uniform_real_distribution<float> random(-.5f,.5f);
    std::vector<float> kv(N*W),score(N*W),ape(ratio*W),norm(D);
    for(auto& x:kv)x=random(rng);
    for(auto& x:score)x=random(rng)*4;
    for(auto& x:ape)x=random(rng);
    for(auto& x:norm)x=1+random(rng)*.25f;
    // Deliberately distinct overlap halves; the startup CSA block has no
    // negative-position rows and must use only its second projection half.
    Buffer<float> dkv(kv),dscore(score),dape(ape),dnorm(norm);
    const std::vector<float> initial(state*W+GUARD,sentinel);
    const std::vector<uint16_t> blank((nc+1)*D+GUARD,0x7bff);
    Buffer<float> fullkv(initial),fullsc(initial),splitkv(initial),splitsc(initial);
    Buffer<uint16_t> full(blank),splitcache(blank);
    auto invoke=[&](Buffer<float>& rk,Buffer<float>& rs,Buffer<uint16_t>& cache,int pos,int count) {
        ds::BatchComp b{};b.c.ring_kv=rk.p;b.c.ring_sc=rs.p;b.c.cache=cache.p;
        b.c.ape=dape.p;b.c.norm=dnorm.p;b.c.D=D;b.c.ratio=ratio;b.c.overlap=overlap;
        b.kv=dkv.p+pos*W;b.sc=dscore.p+pos*W;
        ds::compress_batch(&b,1,pos,count,64,rope,1e-6f,nullptr);
    };
    invoke(fullkv,fullsc,full,0,N);
    if(p0)invoke(splitkv,splitsc,splitcache,0,p0);
    const auto before=splitcache.get();
    invoke(splitkv,splitsc,splitcache,p0,T);ck(cudaDeviceSynchronize());
    const auto actual=full.get(),split=splitcache.get();
    need(actual==split,"compressor full/split cache byte identity");
    need(fullkv.get()==splitkv.get() && fullsc.get()==splitsc.get(),"compressor full/split ring identity");
    if(nc==p0/ratio)need(before==split,"incomplete block changed compressor cache");
    for(size_t i=nc*D;i<actual.size();++i)need(actual[i]==0x7bff,"compressor unwritten row/guard");
    const auto ringkv=fullkv.get(),ringsc=fullsc.get();
    auto expected_kv=initial,expected_sc=initial;
    for(int pos=std::max(0,N-state);pos<N;++pos)for(int c=0;c<W;++c) {
        expected_kv[(pos%state)*W+c]=kv[pos*W+c];
        expected_sc[(pos%state)*W+c]=score[pos*W+c]+ape[(pos%ratio)*W+c];
    }
    need(ringkv==expected_kv && ringsc==expected_sc,"compressor chronological ring and score+bias");
    std::vector<double> reference(nc*D),rounded(nc*D);
    for(int b=0;b<nc;++b) {
        const int start=b*ratio,first=overlap?std::max(0,start-ratio):start;
        std::vector<double> pooled(D);double sq=0;
        for(int d=0;d<D;++d) {
            double mx=-INFINITY;
            for(int pos=first;pos<start+ratio;++pos) {
                const int c=(overlap && pos>=start?D:0)+d;
                // Inputs to softmax are the stored FP32 score+bias sums.
                mx=std::max(mx,double(score[pos*W+c]+ape[(pos%ratio)*W+c]));
            }
            double den=0,num=0;
            for(int pos=first;pos<start+ratio;++pos) {
                const int c=(overlap && pos>=start?D:0)+d;
                const double e=std::exp(double(score[pos*W+c]+ape[(pos%ratio)*W+c])-mx);
                den+=e;num+=e*kv[pos*W+c];
            }
            pooled[d]=num/den;sq+=pooled[d]*pooled[d];
        }
        const double inv=1/std::sqrt(sq/D+1e-6);
        for(int d=0;d<D;++d)pooled[d]*=inv*norm[d];
        rotate_host(pooled,rope,start,true);
        for(int d=0;d<D;++d) {
            reference[b*D+d]=pooled[d];rounded[b*D+d]=hf(fh(float(pooled[d])));
        }
    }
    // Compare normalized/rotated cache values before and after the unavoidable
    // F16 storage rounding separately. No adjustment to the model KL gate.
    std::vector<float> decoded(nc*D);
    for(int i=0;i<nc*D;++i)decoded[i]=hf(actual[i]);
    const auto math=compare(decoded,reference),storage=compare(decoded,rounded);
    need(math.rel<5e-4 && math.max_abs<.004,"compressor FP64 plus F16-storage screen");
    need(storage.rel<2e-4 && storage.max_abs<.004,"compressor rounded FP64 screen");
    std::printf("{\"component\":\"compressor\",\"ratio\":%d,\"overlap\":%s,\"p0\":%d,\"T\":%d,"
                "\"completed_rows\":%d,\"fp64_relative_l2\":%.9g,\"fp64_max_abs\":%.9g,"
                "\"rounded_fp64_relative_l2\":%.9g,\"rounded_fp64_max_abs\":%.9g,"
                "\"full_split_byte_identity\":true,\"ring_identity\":true,\"pass\":true}\n",
                ratio,overlap?"true":"false",p0,T,nc,math.rel,math.max_abs,storage.rel,storage.max_abs);
    return std::vector<uint16_t>(actual.begin(),actual.begin()+(nc+1)*D);
}

// Real model dimensions and the same cuBLAS T/N, strides and F16 conversion
// points as glm_prefill.cu. Sparse deterministic weights make an independent
// FP64 oracle affordable while exercising full dense backend shapes.
struct OutputProjection {
    static constexpr int GROUPS=8,RANK=1024,GIN=H*D/GROUPS,OL=GROUPS*RANK,E=4096,NONZERO=8;
    Buffer<uint16_t> wa,wb;
    cublasHandle_t handle{};
    static int index(int row,int s,int k) {return (row*17+s*(k==GIN?509:997))%k;}
    static float weight(int row,int s) {return ((row+s)&1?-1.f:1.f)*float(1+(row*3+s)%4)/16;}
    static std::vector<uint16_t> weights(int rows,int k) {
        std::vector<uint16_t> out(size_t(rows)*k,0);
        for(int r=0;r<rows;++r)for(int s=0;s<NONZERO;++s)out[size_t(r)*k+index(r,s,k)]=fh(weight(r,s));
        return out;
    }
    OutputProjection():wa(weights(GROUPS*RANK,GIN)),wb(weights(E,OL)) {
        need(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS,"create projection BLAS");
    }
    ~OutputProjection(){cublasDestroy(handle);}
    void gemm(const uint16_t* w,const uint16_t* x,float* y,int n,int k,int ldx,int ldy) {
        const float one=1,zero=0;
        need(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,2,k,&one,w,CUDA_R_16F,k,
                         x,CUDA_R_16F,ldx,&zero,y,CUDA_R_32F,ldy,CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS,"projection GEMM");
    }
    void check(const float* candidate,const float* baseline,const std::vector<float>& host_candidate,
               const std::vector<double>& host_baseline,int ratio,int p0,int t) {
        // Candidate and baseline are two columns of one batch. This exercises
        // HD/OL/E leading dimensions as well as the group offsets.
        Buffer<float> packed(std::vector<float>(2*H*D+GUARD,sentinel));
        ck(cudaMemcpy(packed.p,candidate,H*D*sizeof(float),cudaMemcpyDeviceToDevice));
        ck(cudaMemcpy(packed.p+H*D,baseline,H*D*sizeof(float),cudaMemcpyDeviceToDevice));
        Buffer<uint16_t> x(std::vector<uint16_t>(2*H*D+GUARD,0x7bff)),a16(std::vector<uint16_t>(2*OL+GUARD,0x7bff));
        Buffer<float> a(std::vector<float>(2*OL+GUARD,sentinel)),b(std::vector<float>(2*E+GUARD,sentinel));
        ds::to_f16(packed.p,x.p,2*H*D,nullptr);
        for(int g=0;g<GROUPS;++g)gemm(wa.p+size_t(g)*RANK*GIN,x.p+g*GIN,a.p+g*RANK,RANK,GIN,H*D,OL);
        ds::to_f16(a.p,a16.p,2*OL,nullptr);gemm(wb.p,a16.p,b.p,E,OL,OL,E);
        ck(cudaDeviceSynchronize());
        const auto xv=x.get(),av16=a16.get();const auto av=a.get(),bv=b.get();
        for(int token=0;token<2;++token)for(int i=0;i<H*D;++i)
            need(xv[token*H*D+i]==fh(float(token?host_baseline[i]:host_candidate[i])),"pre-wo_a FP16 conversion");
        for(size_t i=2*H*D;i<xv.size();++i)need(xv[i]==0x7bff,"projection input guard");
        for(int i=0;i<2*OL;++i)need(av16[i]==fh(av[i]),"pre-wo_b FP16 conversion");
        for(size_t i=2*OL;i<av16.size();++i)need(av16[i]==0x7bff,"projection intermediate guard");
        std::vector<double> ar(2*OL),br(2*E);
        for(int token=0;token<2;++token) {
            for(int r=0;r<OL;++r)for(int z=0;z<NONZERO;++z)
                ar[token*OL+r]+=double(weight(r,z))*hf(xv[token*H*D+(r/RANK)*GIN+index(r,z,GIN)]);
            for(int r=0;r<E;++r)for(int z=0;z<NONZERO;++z)
                br[token*E+r]+=double(weight(r,z))*hf(av16[token*OL+index(r,z,OL)]);
        }
        const auto ea=compare(av,ar),eb=compare(bv,br);
        need(ea.rel<1e-5 && eb.rel<1e-5,"grouped projection FP64 backend parity");
        const std::vector<double> base_ref(bv.begin()+E,bv.begin()+2*E);
        const std::vector<float> got(bv.begin(),bv.begin()+E);
        const auto e=compare(got,base_ref);
        // Includes two explicit F16 rounding boundaries. Component-only screen;
        // full-model distribution/semantic acceptance is still required.
        need(e.rel<1e-3,"post-projection candidate versus fallback screen");
        std::printf("{\"component\":\"grouped_output_projection\",\"ratio\":%d,\"p0\":%d,\"t\":%d,"
                    "\"groups\":8,\"rank\":1024,\"embedding\":4096,\"nonzeros_per_row\":8,"
                    "\"post_wo_relative_l2\":%.9g,\"post_wo_max_abs\":%.9g,\"post_wo_p99_abs\":%.9g,"
                    "\"wo_a_backend_relative_l2\":%.9g,\"wo_b_backend_relative_l2\":%.9g,"
                    "\"batch_columns\":2,\"synthetic_weights\":true,\"pass\":true}\n",
                    ratio,p0,t,e.rel,e.max_abs,e.p99_abs,ea.rel,eb.rel);
    }
};
#endif
