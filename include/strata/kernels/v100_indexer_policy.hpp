#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels::dsv4::p1 {
enum class Mode { automatic, f32, cublas };
// Legacy override wins even over a bad or unsupported experimental request.
inline Mode parse(const char* value,const char* legacy) {
    if(legacy && std::atoi(legacy)==1)return Mode::f32;
    if(!value || !*value || !std::strcmp(value,"auto"))return Mode::automatic;
    if(!std::strcmp(value,"f32"))return Mode::f32;
    if(!std::strcmp(value,"cublas"))return Mode::cublas;
    if(!std::strcmp(value,"volta"))throw std::runtime_error(
        "ABYSS_V100_IDX=volta is unavailable: isolated MMA candidate is not qualified for runtime integration");
    throw std::runtime_error("ABYSS_V100_IDX must be auto, f32 or experimental cublas");
}
inline Mode mode() {
    static const Mode value=parse(std::getenv("ABYSS_V100_IDX"),std::getenv("STRATA_DS4_IDX"));
    return value;
}
// Experimental explicit mode only; auto is deliberately unchanged. Restrict to
// measured long-context regime and actual mixer T. This is not a release threshold.
inline bool eligible(int heads,int dim,int top,int T,int p0) {
    return heads==64 && dim==128 && top==512 && T==2048 && p0>=98304 && p0<=120000;
}

// This is also the actual prefill host guard: tests cover unsupported backends
// without pretending a policy unit test executes inference on those devices.
inline const char* backend_error(Mode selected,bool hip,int major,int minor,int heads,int dim,int top) {
    if(selected!=Mode::cublas)return nullptr;
    if(hip)return "ABYSS_V100_IDX=cublas requires CUDA SM70; HIP is unsupported";
    if(major!=7 || minor!=0)return "ABYSS_V100_IDX=cublas requires an SM70 device";
    if(heads!=64 || dim!=128 || top!=512)
        return "ABYSS_V100_IDX=cublas requires 64 heads, dimension 128 and top-512";
    return nullptr;
}
inline bool use_cublas(Mode selected,int ratio,int heads,int dim,int top,int T,int p0) {
    return selected==Mode::cublas && ratio==4 && eligible(heads,dim,top,T,p0);
}
}
