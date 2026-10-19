#pragma once
// Opt-in P0 host dispatch ledger, never a device-execution or DRAM-traffic counter.
// Set ABYSS_V100_DISPATCH=1 and ABYSS_V100_PROFILE=1 in a separate diagnostic run.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace strata::v100dispatch {
inline bool enabled() {
    static const bool on=[] { const char* v=std::getenv("ABYSS_V100_DISPATCH");return v && std::atoi(v)!=0; }();
    return on;
}
// phase 0: setup/untagged, 1: prefill chunk, 2: decode service/issue.
// position is a chunk anchor for prefill, an exact position for decode issue,
// and -1 for service work (join request_seq to request_issue, never guess).
struct Context { int phase=0,layer=-1,chunk_tokens=0; int64_t position=-1; uint64_t request_seq=0; };
inline Context& current() { static thread_local Context c;return c; }
struct Scope {
    Context previous; bool on;
    explicit Scope(Context c): previous(current()),on(enabled()) { if(on) current()=c; }
    ~Scope() { if(on) current()=previous; }
};
using Key=std::tuple<int,int,int,int64_t,uint64_t,int,int,std::string>;
struct Counts { uint64_t calls=0,rows=0,logical_weight_bytes=0; };
struct Ledger {
    std::mutex mutex; std::map<Key,Counts> records; uint64_t dropped=0;
    ~Ledger() {
        for(const auto& entry:records) {
            const auto& k=entry.first;const auto& v=entry.second;
            std::fprintf(stderr,"V100_DISPATCH {\"phase\":%d,\"layer\":%d,\"chunk_tokens\":%d,"
                "\"position\":%lld,\"request_seq\":%llu,\"type\":%d,\"nt\":%d,\"path\":\"%s\","
                "\"calls\":%llu,\"rows\":%llu,\"logical_weight_bytes\":%llu}\n",
                std::get<0>(k),std::get<1>(k),std::get<2>(k),(long long)std::get<3>(k),
                (unsigned long long)std::get<4>(k),std::get<5>(k),std::get<6>(k),std::get<7>(k).c_str(),
                (unsigned long long)v.calls,(unsigned long long)v.rows,(unsigned long long)v.logical_weight_bytes);
        }
        std::fprintf(stderr,"V100_DISPATCH_END {\"keys\":%zu,\"dropped_calls\":%llu,"
            "\"semantics\":\"host_calls_not_device_execution_or_physical_traffic\"}\n",
            records.size(),(unsigned long long)dropped);
    }
};
inline Ledger& ledger() { static Ledger l;return l; }
inline void record(const char* path,int type,int nt,uint64_t rows=0,uint64_t logical_bytes=0) {
    if(!enabled()) return;
    const auto c=current();auto& l=ledger();std::lock_guard<std::mutex> guard(l.mutex);
    Key key{c.phase,c.layer,c.chunk_tokens,c.position,c.request_seq,type,nt,path};
    auto it=l.records.find(key);
    if(it==l.records.end()) {
        if(l.records.size()>=131072) { ++l.dropped;return; }
        it=l.records.emplace(std::move(key),Counts{}).first;
    }
    ++it->second.calls;it->second.rows+=rows;it->second.logical_weight_bytes+=logical_bytes;
}
// Diagnostic host mirror of the actual sorted gather IDs. Lifetime is one
// layer/chunk invocation, so repeats of a prompt never share a distinct set.
struct InputRows {
    std::vector<int> tokens;
    std::vector<unsigned char> seen;
    uint64_t unique=0;
    bool count(int offset,int n,int type) {
        if(offset<0 || n<0 || (size_t)offset>tokens.size() ||
           (size_t)n>tokens.size()-(size_t)offset) return false;
        std::vector<unsigned char> local(seen.size(),0);
        int distinct=0;
        for(int i=0;i<n;++i) {
            const int t=tokens[(size_t)offset+i];
            if(t<0 || (size_t)t>=seen.size()) return false;
            if(!local[(size_t)t]) { local[(size_t)t]=1;++distinct; }
        }
        for(size_t t=0;t<local.size();++t)
            if(local[t] && !seen[t]) { seen[t]=1;++unique; }
        // A histogram preserves exact distinct rows for each call shape even
        // when identical layer/chunk anchors recur in multiple requests.
        const std::string path="mmq_input_distinct:"+std::to_string(distinct);
        record(path.c_str(),type,n,(uint64_t)distinct);
        return true;
    }
};
} // namespace strata::v100dispatch
