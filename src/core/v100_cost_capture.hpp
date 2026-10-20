#pragma once
// Compile-only P3 cost observation. NVTX host ranges only: no CUDA event,
// synchronization, estimator, allocation decision or arithmetic change.
#ifdef STRATA_V100_CAPTURE_COST
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
namespace strata::v100cost {
struct Ledger {
    std::atomic<unsigned> ranges{0},dropped{0};
    ~Ledger(){std::fprintf(stderr,"V100_COST_END ranges=%u dropped=%u\n",ranges.load(),dropped.load());}
};
inline Ledger& ledger(){static Ledger value;return value;}
inline bool reserve(){
    auto& l=ledger();if(l.ranges.fetch_add(1,std::memory_order_relaxed)<32768)return true;
    l.dropped.fetch_add(1,std::memory_order_relaxed);return false;
}
inline void route(int device,int layer,uint64_t sequence,unsigned cpu,unsigned fetch,unsigned miss,size_t blob){
    if(!reserve())return;
    char label[224];
    std::snprintf(label,sizeof label,"P3_ROUTE device=%d layer=%d seq=%llu cpu=%u fetch=%u miss=%u blob=%llu",
        device,layer,(unsigned long long)sequence,cpu,fetch,miss,(unsigned long long)blob);
    nvtxMarkA(label);
}
struct Scope {
    bool active=false;
    Scope(const char* kind,int device,int layer,uint64_t sequence,int experts=-1){
        if(!reserve())return;
        char label[160];
        std::snprintf(label,sizeof label,"P3_COST kind=%s device=%d layer=%d seq=%llu experts=%d",
                      kind,device,layer,(unsigned long long)sequence,experts);
        nvtxRangePushA(label);active=true;
    }
    ~Scope(){if(active)nvtxRangePop();}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
};
}
#endif
