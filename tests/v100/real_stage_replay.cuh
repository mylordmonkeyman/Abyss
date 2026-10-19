// Loaded only by the CUDA-only latent diagnostic. Uses actual model projections,
// cache rows and selections captured by STRATA_V100_CAPTURE_STAGES.
#pragma once
#ifndef STRATA_USE_HIP
template<class T> std::vector<T> tensor_file(const std::filesystem::path& p,size_t count) {
    need(std::filesystem::file_size(p)==count*sizeof(T),"real-stage tensor byte size");
    std::vector<T> v(count);std::ifstream f(p,std::ios::binary);
    if(count)f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));
    need(bool(f),"read real-stage tensor");return v;
}
void real_stage_case(const std::filesystem::path& root,int expected_ratio) {
    const auto dir=root/("ratio-"+std::to_string(expected_ratio));
    need(std::filesystem::exists(dir/"complete"),"real-stage capture incomplete");
    std::ifstream meta(dir/"meta.txt");std::string magic;
    int layer,p0,T,t,hist,ratio,comp_rows,heads,dim,nrot;ds::Rope rope{};float eps;
    meta>>magic>>layer>>p0>>T>>t>>hist>>ratio>>comp_rows>>heads>>dim>>nrot
        >>rope.theta_scale>>rope.freq_scale>>rope.ext_factor>>rope.attn_factor>>rope.corr_lo>>rope.corr_hi>>eps;
    need(bool(meta) && magic=="ABYSS_V100_STAGE_1","real-stage metadata ABI");
    need(ratio==expected_ratio && heads==H && dim==D && nrot==64 && T>0 && T<=256
         && t>=0 && t<T && p0+t==2048 && hist==std::min(p0,127),"real-stage geometry");
    need(comp_rows==(ratio?(p0+T)/ratio:0),"real-stage completed rows");
    const int p=p0+t,nr=std::min(p+1,WINDOW),visible=ratio?(p+1)/ratio:0;
    const int nc=ratio==4?std::min(TOP,visible):visible,n=nr+nc;
    auto q=tensor_file<float>(dir/"q.f32",H*D),qraw=tensor_file<float>(dir/"q_raw.f32",H*D),
         kvraw=tensor_file<float>(dir/"kv_raw.f32",D),norm=tensor_file<float>(dir/"kv_norm.f32",D),
         sinks=tensor_file<float>(dir/"sinks.f32",H),baseline=tensor_file<float>(dir/"attention.f32",H*D);
    auto keys=tensor_file<uint16_t>(dir/"keys.f16",(hist+T)*D),comp=tensor_file<uint16_t>(dir/"comp.f16",comp_rows*D);
    auto selection=ratio==4?tensor_file<int>(dir/"selection.i32",TOP):std::vector<int>(TOP,0);
    if(comp.empty())comp.resize(D,fh(NAN));
    // Independent query RMS/RoPE and raw-KV RMS*weight/RoPE references use the
    // actual learned projection outputs, before their preparation kernels.
    std::vector<double> query_ref(H*D),kv_ref(D),kv_rounded(D);
    for(int h=0;h<H;++h) {
        double ss=0;for(int d=0;d<D;++d)ss+=double(qraw[h*D+d])*qraw[h*D+d];
        const double inv=1/std::sqrt(ss/D+eps);std::vector<double> head(D);
        for(int d=0;d<D;++d)head[d]=qraw[h*D+d]*inv;
        rotate_host(head,rope,p,true);std::copy(head.begin(),head.end(),query_ref.begin()+h*D);
    }
    double ss=0;for(auto v:kvraw)ss+=double(v)*v;
    const double inv=1/std::sqrt(ss/D+eps);
    for(int d=0;d<D;++d)kv_ref[d]=kvraw[d]*inv*norm[d];
    rotate_host(kv_ref,rope,p,true);
    std::vector<float> stored_kv(D);
    for(int d=0;d<D;++d) {stored_kv[d]=hf(keys[(hist+t)*D+d]);kv_rounded[d]=hf(fh(float(kv_ref[d])));}
    const auto eqprep=compare(q,query_ref),ekv=compare(stored_kv,kv_ref),ekvr=compare(stored_kv,kv_rounded);
    need(eqprep.rel<1e-5,"real query preparation FP64 parity");
    need(ekv.rel<5e-4 && ekvr.rel<2e-4,"real raw KV preparation and F16 storage parity");
    Buffer<float> dq(q),dsink(sinks),scores(std::vector<float>(H*n+GUARD,sentinel)),
                  prob(std::vector<float>(H*n+GUARD,sentinel)),den(std::vector<float>(H+GUARD,sentinel)),
                  pv(std::vector<float>(H*D+GUARD,sentinel)),output(std::vector<float>(H*D+GUARD,sentinel));
    Buffer<uint16_t> dk(keys),dc(comp),gathered(std::vector<uint16_t>(n*D+GUARD,0x7bff));
    Buffer<int> sel(selection),ids(std::vector<int>(n+GUARD,-12345)),status(std::vector<int>(1,0));
    gather<<<n,D>>>(dk.p,dc.p,sel.p,hist,p0,t,ratio,comp_rows,gathered.p,ids.p,status.p);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(status.get()[0]==0,"real model selection rejected");
    const auto gv=gathered.get();const auto row_ids=ids.get();std::vector<const uint16_t*> rows;
    for(int j=0;j<n;++j) {
        const int r=j<nr?p-nr+1+j-(p0-hist):ratio==4?selection[j-nr]:j-nr;
        need(j<nr || (r>=0 && r<visible),"real compressed causality");
        const auto* src=(j<nr?keys.data():comp.data())+r*D;rows.push_back(src);
        need(std::equal(src,src+D,gv.begin()+j*D),"real gather byte identity");
        need(row_ids[j]==(j<nr?p-nr+1+j:-r-1),"real gather absolute identity");
    }
    for(size_t i=n*D;i<gv.size();++i)need(gv[i]==0x7bff,"real gather key guard");
    for(size_t i=n;i<row_ids.size();++i)need(row_ids[i]==-12345,"real gather ID guard");
    auto launch=[&](cudaStream_t stream) {
        qk_mma<<<dim3(H/8,(n+31)/32),32,0,stream>>>(dq.p,gathered.p,scores.p,n);
        softmax_sink<<<1,H,0,stream>>>(scores.p,dsink.p,prob.p,den.p,n);
        pv_mma<<<dim3(H/8,D/32),32,0,stream>>>(prob.p,gathered.p,pv.p,n,1024);
        inverse_rope<<<H,D,0,stream>>>(pv.p,output.p,rope,p);
    };
    launch(nullptr);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(stream);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int i=0;i<3;++i)ck(cudaGraphLaunch(executable,stream));
    ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    std::vector<double> qk(H*n),pr(H*n),dn(H),latent(H*D),rotated(H*D),f32(baseline.begin(),baseline.end());
    for(int h=0;h<H;++h) {
        double mx=sinks[h];
        for(int j=0;j<n;++j) {
            double dot=0;for(int d=0;d<D;++d)dot+=double(q[h*D+d])*hf(rows[j][d]);
            qk[h*n+j]=dot;mx=std::max(mx,dot/std::sqrt(double(D)));
        }
        dn[h]=std::exp(sinks[h]-mx);
        for(int j=0;j<n;++j)dn[h]+=std::exp(qk[h*n+j]/std::sqrt(double(D))-mx);
        std::vector<double> z(D);
        for(int j=0;j<n;++j) {
            pr[h*n+j]=std::exp(qk[h*n+j]/std::sqrt(double(D))-mx)/dn[h];
            for(int d=0;d<D;++d)z[d]+=pr[h*n+j]*hf(rows[j][d]);
        }
        std::copy(z.begin(),z.end(),latent.begin()+h*D);rotate_host(z,rope,p,false);
        std::copy(z.begin(),z.end(),rotated.begin()+h*D);
    }
    auto stage=[&](const char* name,const Error& e) {
        std::printf("{\"real_stage\":\"%s\",\"layer\":%d,\"ratio\":%d,\"position\":%d,"
                    "\"relative_l2\":%.9g,\"max_abs\":%.9g,\"p99_abs\":%.9g,\"pass\":true}\n",
                    name,layer,ratio,p,e.rel,e.max_abs,e.p99_abs);
    };
    const auto eq=compare(scores.get(),qk),ep=compare(prob.get(),pr),ed=compare(den.get(),dn),
               ev=compare(pv.get(),latent),eo=compare(output.get(),rotated),ef=compare(output.get(),f32);
    need(eq.rel<1e-5 && ep.rel<1e-5 && ed.rel<1e-5,"real QK/softmax screen");
    need(ev.rel<1e-5 && eo.rel<1e-5 && ef.rel<1e-5,"real PV/inverse RoPE screen");
    stage("query_prep",eqprep);stage("raw_kv_with_storage",ekv);stage("raw_kv_rounded_reference",ekvr);
    stage("qk",eq);stage("probability",ep);stage("sink_denominator",ed);
    stage("pv",ev);stage("inverse_rope",eo);stage("versus_captured_fp32",ef);
}
void real_stages(const std::filesystem::path& root) {
    for(int ratio:{0,4,128})real_stage_case(root,ratio);
    std::puts("{\"suite\":\"real_model_stage_replay\",\"cases\":3,\"production_enabled\":false,\"full_model_kl_gate_complete\":false,\"pass\":true}");
}
#endif
