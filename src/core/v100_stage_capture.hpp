// Internal, compile-only diagnostic. No runtime/public capture control.
#pragma once
#ifdef STRATA_V100_CAPTURE_STAGES
#include <filesystem>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace strata::core::v100diag {
struct StageCapture {
    bool enabled=false;
    int t=0,comp_rows=0;
    std::filesystem::path directory;
    cudaStream_t stream{};
    StageCapture(int layer,int p0,int T,int hist,int ratio,int heads,int dim,int nrot,
                 const strata::kernels::dsv4::Rope& rope,float eps,cudaStream_t s):stream(s) {
        static bool seen[3]={false,false,false};
        const int slot=ratio==0?0:ratio==4?1:ratio==128?2:-1;
        constexpr int target=2048;
        if(slot<0 || seen[slot] || p0>target || p0+T<=target)return;
        if(T>256 || heads!=64 || dim!=512 || nrot!=64 || hist>127)
            throw std::runtime_error("unsupported V100 capture geometry");
        t=target-p0;comp_rows=ratio?(p0+T)/ratio:0;
        directory=std::filesystem::path("v100-stage-capture")/("ratio-"+std::to_string(ratio));
        std::filesystem::create_directories(directory.parent_path());
        if(!std::filesystem::create_directory(directory))
            throw std::runtime_error("V100 capture refuses existing stage directory");
        std::ofstream f(directory/"meta.txt");
        f<<"ABYSS_V100_STAGE_1\n"<<layer<<' '<<p0<<' '<<T<<' '<<t<<' '<<hist<<' '
         <<ratio<<' '<<comp_rows<<' '<<heads<<' '<<dim<<' '<<nrot<<'\n'<<std::setprecision(9)
         <<rope.theta_scale<<' '<<rope.freq_scale<<' '<<rope.ext_factor<<' '<<rope.attn_factor<<' '
         <<rope.corr_lo<<' '<<rope.corr_hi<<' '<<eps<<'\n';
        if(!f)throw std::runtime_error("V100 capture metadata write failed");
        seen[slot]=true;enabled=true;
        std::fprintf(stderr,"v100 stage capture: layer %d ratio %d position %d\n",layer,ratio,target);
    }
    template<class T> void copy(const char* name,const T* source,size_t count) {
        if(!enabled)return;
        static_assert(sizeof(T)==2 || sizeof(T)==4);
        // Across all element types and layers, all files together stay bounded.
        const size_t bytes=count*sizeof(T);
        reserve(bytes);
        if(cudaStreamSynchronize(stream)!=cudaSuccess)
            throw std::runtime_error("V100 capture stream synchronization failed");
        std::vector<T> host(count);
        if(bytes && cudaMemcpy(host.data(),source,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)
            throw std::runtime_error("V100 capture device read failed");
        std::ofstream f(directory/name,std::ios::binary);
        if(bytes)f.write(reinterpret_cast<const char*>(host.data()),bytes);
        if(!f)throw std::runtime_error("V100 capture tensor write failed");
    }
    static void reserve(size_t bytes) {
        static size_t total=0;
        if(bytes>16*1024*1024 || total+bytes>32*1024*1024)
            throw std::runtime_error("V100 capture exceeds hard byte cap");
        total+=bytes;
    }
    void finish() {
        if(!enabled)return;
        std::ofstream f(directory/"complete");f<<"diagnostic capture; not performance evidence\n";
        if(!f)throw std::runtime_error("V100 capture completion write failed");
    }
};
}
#endif
