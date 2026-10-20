#pragma once
#ifdef STRATA_V100_CAPTURE_INDEXER
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
namespace strata::core::v100diag {
// Sample the same five live query rows as the score/selection capture. CPU
// storage only; compile-only diagnostic synchronization is excluded from timing.
inline void mma_query_range(IndexerCapture& capture,const float* queries){
    if(!capture.enabled)return;
    IndexerCapture::ck(cudaStreamSynchronize(capture.stream));
    size_t zeros=0,nonzero=0,heads=0;double minimum=std::numeric_limits<double>::infinity(),maximum=0;
    int min_shift=1000,max_shift=-1000;
    int min_token_shift=1000,max_token_shift=-1000;
    for(int t:capture.samples){
        const auto values=capture.read(queries+(size_t)t*64*128,64*128);
        float token_peak=0;
        for(int h=0;h<64;++h){
            float peak=0;
            for(int d=0;d<128;++d){
                const float v=values[h*128+d];
                if(!std::isfinite(v))throw std::runtime_error("nonfinite live MMA query");
                const double a=std::abs(double(v));peak=std::max(peak,std::abs(v));maximum=std::max(maximum,a);
                if(a==0)++zeros;else{++nonzero;minimum=std::min(minimum,a);}
            }
            int exponent=0;if(peak>0)std::frexp(peak,&exponent);
            min_shift=std::min(min_shift,-exponent);max_shift=std::max(max_shift,-exponent);++heads;
            token_peak=std::max(token_peak,peak);
        }
        int exponent=0;if(token_peak>0)std::frexp(token_peak,&exponent);
        min_token_shift=std::min(min_token_shift,-exponent);max_token_shift=std::max(max_token_shift,-exponent);
    }
    std::ofstream f(capture.dir/"query-range.json");f<<std::setprecision(9)
        <<"{\"sampled_values\":"<<zeros+nonzero<<",\"zero_values\":"<<zeros
        <<",\"sampled_heads\":"<<heads<<",\"minimum_nonzero_abs\":"<<(nonzero?minimum:0)
        <<",\"maximum_abs\":"<<maximum<<",\"minimum_head_normalization_shift\":"<<min_shift
        <<",\"maximum_head_normalization_shift\":"<<max_shift
        <<",\"minimum_token_normalization_shift\":"<<min_token_shift
        <<",\"maximum_token_normalization_shift\":"<<max_token_shift<<"}\n";
    if(!f)throw std::runtime_error("MMA query range write failed");
}
}
#endif
