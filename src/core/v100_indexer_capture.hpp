#pragma once
// Compile-only P1 same-input diagnostic. No production toggle or GPU allocation.
#ifdef STRATA_V100_CAPTURE_INDEXER
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::core::v100diag {
struct IndexerCapture {
    bool enabled=false;
    int T=0,hd=0,ld=0,p0=0;
    cudaStream_t stream{};
    std::filesystem::path dir;
    std::vector<int> candidate_selection;
    const std::vector<int> samples{0,255,256,1023,2047};
    static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    IndexerCapture(bool candidate,int layer,int pos,int tokens,int width,int stride,cudaStream_t s)
        :T(tokens),hd(width),ld(stride),p0(pos),stream(s){
        if(!candidate || pos!=98304 || tokens!=2048)return;
        if(layer<0 || layer>=64 || width!=32768 || stride<(pos+tokens)/4)
            throw std::runtime_error("unsupported P1 live-capture geometry");
        static bool seen[64]={};
        if(seen[layer])throw std::runtime_error("P1 capture target repeated in one process");
        dir=std::filesystem::path("p1-indexer-capture")/("layer-"+std::to_string(layer));
        std::filesystem::create_directories(dir.parent_path());
        if(!std::filesystem::create_directory(dir))throw std::runtime_error("P1 capture refuses existing layer directory");
        std::ofstream f(dir/"meta.txt");
        f<<"ABYSS_P1_LIVE_1\n"<<layer<<' '<<p0<<' '<<T<<' '<<hd<<' '<<ld<<'\n';
        for(int t:samples)f<<t<<' ';f<<'\n';
        if(!f)throw std::runtime_error("P1 capture metadata write failed");
        seen[layer]=true;enabled=true;
    }
    template<class V> std::vector<V> read(const V* source,size_t count){
        std::vector<V> result(count);
        ck(cudaMemcpy(result.data(),source,count*sizeof(V),cudaMemcpyDeviceToHost));return result;
    }
    template<class V> void save(const std::string& name,const V* source,size_t count){
        static_assert(sizeof(V)==4);
        auto values=read(source,count);
        reserve(count*sizeof(V));
        std::ofstream f(dir/name,std::ios::binary);
        f.write(reinterpret_cast<const char*>(values.data()),count*sizeof(V));
        if(!f)throw std::runtime_error("P1 capture tensor write failed");
    }
    static void reserve(size_t bytes){
        static size_t total=0;
        if(bytes>1024*1024 || total+bytes>56*1024*1024)
            throw std::runtime_error("P1 capture exceeds hard byte cap");
        total+=bytes;
    }
    void selected(const float* score,const int* ids,bool candidate){
        if(!enabled)return;
        ck(cudaStreamSynchronize(stream));
        if(candidate)candidate_selection=read(ids,(size_t)T*512);
        for(int t:samples){
            const std::string prefix="t-"+std::to_string(t)+(candidate?"-candidate":"-reference");
            save(prefix+"-score.f32",score+(size_t)t*ld,(p0+t+1)/4);
            save(prefix+"-ids.i32",ids+(size_t)t*512,512);
        }
    }
    void attention(const float* output,bool candidate){
        if(!enabled)return;
        ck(cudaStreamSynchronize(stream));
        for(int t:samples)save("t-"+std::to_string(t)+(candidate?"-candidate":"-reference")+"-attention.f32",output+(size_t)t*hd,hd);
        if(candidate){std::ofstream f(dir/"complete");f<<"same inputs; diagnostic timing only\n";
            if(!f)throw std::runtime_error("P1 capture completion write failed");}
    }
    void restore(int* ids){
        if(!enabled)return;
        if(candidate_selection.size()!=(size_t)T*512)throw std::runtime_error("P1 candidate selection snapshot missing");
        ck(cudaMemcpyAsync(ids,candidate_selection.data(),candidate_selection.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
        ck(cudaStreamSynchronize(stream));
        if(read(ids,candidate_selection.size())!=candidate_selection)
            throw std::runtime_error("P1 candidate selection was not restored bitwise");
        std::ofstream f(dir/"selection-restored");f<<candidate_selection.size()<<" int32 values; bitwise verified\n";
        if(!f)throw std::runtime_error("P1 restoration record write failed");
    }
};
}
#endif
