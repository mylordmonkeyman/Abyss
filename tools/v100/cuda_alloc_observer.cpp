// P0 diagnostic only. Build: c++ -shared -fPIC -std=c++17 -O2 -pthread
//   tools/v100/cuda_alloc_observer.cpp -ldl -o cuda_alloc_observer.so
// Run with ABYSS_CUDA_ALLOC_LOG=/absolute/file.jsonl LD_PRELOAD=... .
// No CUDA headers or engine changes. Requires dynamically interposable cudart;
// verify nonzero events and ELF bindings on the runner before using the ledger.
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
thread_local bool nested = false;
int fd = -1;
bool opened = false;
unsigned long long seq = 0;
struct Guard { bool outer; Guard(): outer(!nested) { nested = true; }
    ~Guard() { if (outer) nested = false; } };
void event(const char* api, const void* ptr, size_t bytes, unsigned flags,
           int result, const void* alias = nullptr) {
    pthread_mutex_lock(&lock);
    if (!opened) {
        opened = true;
        const char* path = std::getenv("ABYSS_CUDA_ALLOC_LOG");
        if (path && *path) fd = open(path, O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC, 0600);
    }
    if (fd >= 0) {
        timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts);
        char buf[512];
        int n = std::snprintf(buf, sizeof buf,
            "{\"seq\":%llu,\"pid\":%ld,\"monotonic_ns\":%llu,\"api\":\"%s\","
            "\"ptr\":%llu,\"bytes\":%llu,\"flags\":%u,\"result\":%d,\"alias\":%llu}\n",
            ++seq, (long)getpid(), (unsigned long long)ts.tv_sec*1000000000+ts.tv_nsec,
            api, (unsigned long long)(uintptr_t)ptr, (unsigned long long)bytes,
            flags, result, (unsigned long long)(uintptr_t)alias);
        if (n > 0 && n < (int)sizeof buf) {
            // A short/failed write invalidates the ledger; parser checks seq.
            ssize_t ignored = write(fd, buf, (size_t)n); (void)ignored;
        }
    }
    pthread_mutex_unlock(&lock);
}
__attribute__((destructor)) void finish() {
    Guard g; if (g.outer && opened) event("observer_end", nullptr, 0, 0, 0);
}
template<class Fn> Fn next(const char* name) { return reinterpret_cast<Fn>(dlsym(RTLD_NEXT,name)); }
}
extern "C" int cudaMalloc(void** p, size_t n) {
    Guard g; auto f=next<int(*)(void**,size_t)>("cudaMalloc"); int r=f?f(p,n):999;
    if(g.outer) event("cudaMalloc",r?nullptr:*p,n,0,r); return r;
}
extern "C" int cudaMallocHost(void** p, size_t n) {
    Guard g; auto f=next<int(*)(void**,size_t)>("cudaMallocHost"); int r=f?f(p,n):999;
    if(g.outer) event("cudaMallocHost",r?nullptr:*p,n,0,r); return r;
}
extern "C" int cudaHostAlloc(void** p, size_t n, unsigned flags) {
    Guard g; auto f=next<int(*)(void**,size_t,unsigned)>("cudaHostAlloc"); int r=f?f(p,n,flags):999;
    if(g.outer) event("cudaHostAlloc",r?nullptr:*p,n,flags,r); return r;
}
extern "C" int cudaHostRegister(void* p, size_t n, unsigned flags) {
    Guard g; auto f=next<int(*)(void*,size_t,unsigned)>("cudaHostRegister"); int r=f?f(p,n,flags):999;
    if(g.outer) event("cudaHostRegister",p,n,flags,r); return r;
}
extern "C" int cudaFree(void* p) {
    Guard g; auto f=next<int(*)(void*)>("cudaFree"); int r=f?f(p):999;
    if(g.outer) event("cudaFree",p,0,0,r); return r;
}
extern "C" int cudaFreeHost(void* p) {
    Guard g; auto f=next<int(*)(void*)>("cudaFreeHost"); int r=f?f(p):999;
    if(g.outer) event("cudaFreeHost",p,0,0,r); return r;
}
extern "C" int cudaHostUnregister(void* p) {
    Guard g; auto f=next<int(*)(void*)>("cudaHostUnregister"); int r=f?f(p):999;
    if(g.outer) event("cudaHostUnregister",p,0,0,r); return r;
}
extern "C" int cudaHostGetDevicePointer(void** d, void* h, unsigned flags) {
    Guard g; auto f=next<int(*)(void**,void*,unsigned)>("cudaHostGetDevicePointer"); int r=f?f(d,h,flags):999;
    if(g.outer) event("cudaHostGetDevicePointer",h,0,flags,r,r?nullptr:*d); return r;
}
