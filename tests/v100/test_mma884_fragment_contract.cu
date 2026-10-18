// Four independent 8x8 products/warp; host FP64 reference and scalar device reference.
#include "volta_mma884.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void need(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
template<class T> struct Buffer {
    T* p = nullptr; size_t n;
    explicit Buffer(const std::vector<T>& x): n(x.size()) { ck(cudaMalloc((void**) &p, n*sizeof(T))); ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    std::vector<T> get() const { std::vector<T> x(n); ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return x; }
};
__global__ void contract(const uint16_t* A, const uint16_t* B, const float* seed, float* out,
                         int K, int rows, int cols, int permutation) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    const int lane = threadIdx.x;
    const int qp = (lane >> 2) & 3, source = (qp + permutation) % 4;
    const int j = (lane & 3) + 4 * (lane >> 4);
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = (lane & 1) + 4*(lane >> 4) + 2*((i >> 1) & 1);
        const int c = (lane & 2) + 4*(i >> 2) + (i & 1);
        acc[i] = seed[qp*64 + r*8 + c];
    }
    for (int k0 = 0; k0 < K; k0 += 4) {
        uint32_t a[2] = {}, b[2] = {};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint16_t av = j < rows && k0+k < K ? A[(source*8+j)*K+k0+k] : 0;
            const uint16_t bv = j < cols && k0+k < K ? B[(source*K+k0+k)*8+j] : 0;
            a[k/2] |= (uint32_t) av << (16*(k%2)); b[k/2] |= (uint32_t) bv << (16*(k%2));
        }
        strata::kernels::volta::mma884(acc, a[0], a[1], b[0], b[1]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = (lane & 1) + 4*(lane >> 4) + 2*((i >> 1) & 1);
        const int c = (lane & 2) + 4*(i >> 2) + (i & 1);
        out[qp*64+r*8+c] = acc[i];
    }
#endif
}
__global__ void scalar(const uint16_t* A, const uint16_t* B, const float* seed, float* out,
                       int K, int rows, int cols, int permutation) {
    const int i = threadIdx.x, qp = i/64, r = (i%64)/8, c = i%8, source = (qp+permutation)%4;
    float sum = seed[i];
    if (r < rows && c < cols) for (int k = 0; k < K; ++k)
        sum += __half2float(__ushort_as_half(A[(source*8+r)*K+k])) * __half2float(__ushort_as_half(B[(source*K+k)*8+c]));
    out[i] = sum;
}

void run_case(int K, int rows, int cols, int permutation, int mode) {
    std::vector<uint16_t> a(4*8*K), b(4*K*8);
    std::vector<float> seed(256), blank(256+32,12345.0f);
    for (size_t i=0;i<a.size();++i) {
        float v = (int)((i*13+i/K)%31)-15;
        v /= 16;
        if (mode==1) v=0;
        if (mode==2) v=std::ldexp(1.0f,-24); // FP16 subnormal stress; measured absolute error is reported.
        a[i]=__half_as_ushort(__float2half_rn(v));
    }
    for (size_t i=0;i<b.size();++i) b[i]=__half_as_ushort(__float2half_rn(((int)((i*7+i/8)%29)-14)/16.0f));
    for (int i=0;i<256;++i) seed[i]=mode==2 ? 0 : (i-128)/32.0f;
    Buffer<uint16_t> da(a), db(b); Buffer<float> ds(seed), dout(blank), dref(blank);
    contract<<<1,32>>>(da.p,db.p,ds.p,dout.p,K,rows,cols,permutation); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    scalar<<<1,256>>>(da.p,db.p,ds.p,dref.p,K,rows,cols,permutation); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    // Capture/replay writes every accumulator afresh from seed, exercising repeated launches.
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    contract<<<1,32,0,stream>>>(da.p,db.p,ds.p,dout.p,K,rows,cols,permutation);
    ck(cudaGetLastError()); ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int repeat=0;repeat<4;++repeat) ck(cudaGraphLaunch(exec,stream));
    ck(cudaStreamSynchronize(stream));
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    const auto got=dout.get(), ref=dref.get(); double max_error=0;
    for (int i=0;i<256;++i) {
        const int qp=i/64, r=(i%64)/8, c=i%8, source=(qp+permutation)%4;
        double expected=seed[i];
        if (r<rows && c<cols) for (int k=0;k<K;++k)
            expected+=(double)__half2float(__ushort_as_half(a[(source*8+r)*K+k]))*__half2float(__ushort_as_half(b[(source*K+k)*8+c]));
        need(std::isfinite(got[i]),"nonfinite MMA result");
        max_error=std::max(max_error,std::abs(got[i]-expected));
        need(std::abs(got[i]-expected)<2e-5,"MMA884 FP64 coordinate/operand parity");
        need(std::abs(got[i]-ref[i])<2e-5,"MMA884 device scalar parity");
    }
    for (int i=256;i<288;++i) need(got[i]==12345.0f && ref[i]==12345.0f,"MMA output guard");
    std::printf("{\"case\":\"mma884_contract\",\"K\":%d,\"rows\":%d,\"cols\":%d,\"permutation\":%d,\"mode\":%d,\"max_abs\":%.9g,\"pass\":true}\n",K,rows,cols,permutation,mode,max_error);
}
} // namespace
int main() {
    try {
        cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0)); need(p.major==7 && p.minor==0,"MMA884 contract requires actual SM70");
        int seen[256] = {};
        for (int lane=0;lane<32;++lane) for (int i=0;i<8;++i) {
            const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1), c=(lane&2)+4*(i>>2)+(i&1);
            ++seen[((lane>>2)&3)*64+r*8+c];
        }
        for (int n:seen) need(n==1,"fragment coordinate map is not bijective");
        for (int K:{4,8,17,128}) for (int p:{0,1,3}) run_case(K,8,8,p,0);
        run_case(17,5,7,2,0); run_case(4,8,8,0,1); run_case(4,8,8,0,2);
        std::puts("{\"suite\":\"mma884_fragment_contract\",\"pass\":true}"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"MMA884 contract FAILED: %s\n",e.what()); return 1; }
}
