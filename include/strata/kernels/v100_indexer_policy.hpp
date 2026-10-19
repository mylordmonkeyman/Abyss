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
}
