#include "../../src/core/v100_cublas_selection_control.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace ds=strata::kernels::dsv4;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool value){if(!value)throw std::runtime_error("selection-control contract failed");}
template<class T>struct Buffer{
    T* p;size_t n;explicit Buffer(size_t size):n(size){ck(cudaMalloc((void**)&p,n*sizeof(T)));}
    explicit Buffer(const std::vector<T>& x):Buffer(x.size()){ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(p);}std::vector<T> get(){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
int main(){try{
    cudaDeviceProp device{};ck(cudaGetDeviceProperties(&device,0));need(device.major==7&&device.minor==0);
    for(int p0:{2040,2052,8192})for(int mode:{0,1})for(bool membership:{false,true}){
        const int T=3,n=(p0+T)/4,ld=n+7;
        std::vector<float> q(T*64*128),w(T*64),fake(T*ld+32,12345.0f);
        std::vector<uint16_t> keys(n*128);
        for(size_t i=0;i<q.size();++i)q[i]=mode?0:float(int(i%29)-14)/29;
        for(size_t i=0;i<w.size();++i)w[i]=float(int(i%13)-6)/13;
        for(size_t i=0;i<keys.size();++i)keys[i]=__half_as_ushort(__float2half_rn(float(int(i%19)-9)/19));
        for(int t=0;t<T;++t)for(int j=0;j<(p0+t+1)/4;++j)fake[t*ld+j]=float(j%7);
        Buffer<float> dq(q),dw(w),scores(fake),ref(std::vector<float>(T*n+32,12345.0f));Buffer<uint16_t> dk(keys);
        Buffer<int> selected(std::vector<int>(T*512+32,-12345)),reference(std::vector<int>(T*512+32,-12345));
        Buffer<int> scratch(T*512+4);
        ds::indexer_batch(dq.p,dk.p,dw.p,64,4,p0,T,512,ref.p,n,reference.p,nullptr);
        for(int t=0;t<T;++t)ds::indexer_select(scores.p+t*ld,(p0+t+1)/4,512,selected.p+t*512,nullptr);
        const auto prior=selected.get(),baseline=reference.get();
        cudaStream_t stream=nullptr;
        auto call=[&]{strata::core::v100diag::cublas_selection_control(dq.p,dk.p,dw.p,p0,T,ld,scores.p,selected.p,stream,
            ref.p,T*n,scratch.p,scratch.n,membership);};
        call();ck(cudaDeviceSynchronize());const auto result=selected.get(),stats=scratch.get();
        int ordered=0,sets=0,overlap=0;
        for(int t=0;t<T;++t){const int top=std::min(512,(p0+t+1)/4);
            std::vector<int> expected(baseline.begin()+t*512,baseline.begin()+t*512+top);
            bool same=true;int row_overlap=0;
            for(int i=0;i<top;++i){same&=baseline[t*512+i]==prior[t*512+i];
                row_overlap+=std::find(prior.begin()+t*512,prior.begin()+t*512+top,baseline[t*512+i])!=prior.begin()+t*512+top;}
            ordered+=same;sets+=row_overlap==top;overlap+=row_overlap;
            if(membership)std::sort(expected.begin(),expected.end(),[&](int a,int b){return fake[t*ld+a]>fake[t*ld+b]||(fake[t*ld+a]==fake[t*ld+b]&&a<b);});
            for(int i=0;i<top;++i)need(result[t*512+i]==expected[i]);
        }
        need(stats[T*512]==T&&stats[T*512+1]==ordered&&stats[T*512+2]==sets&&stats[T*512+3]==overlap);
        need(scores.get()==fake);for(size_t i=T*512;i<result.size();++i)need(result[i]==-12345);
        const auto before=result;cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamCreate(&stream));call();ck(cudaStreamSynchronize(stream));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));call();ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<2;++i)ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));need(selected.get()==before);
        ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        ck(cudaStreamDestroy(stream));
        std::printf("{\"case\":\"cublas_selection_control\",\"p0\":%d,\"mode\":%d,\"membership_only\":%s,\"pass\":true}\n",p0,mode,membership?"true":"false");
    }
    std::puts("{\"suite\":\"cublas_selection_control\",\"cases\":12,\"pass\":true}");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"selection control FAILED: %s\n",e.what());return 1;}}
