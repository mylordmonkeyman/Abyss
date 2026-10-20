// P2 feasibility only: shared latent K=V, eight heads/query, online sink softmax.
// No production dispatch, allocation, output projection or inverse RoPE.
#pragma once
#ifndef STRATA_USE_HIP
#include "qkpv_math.cuh"
namespace v100diag {
template<int TILE,int WARPS> struct AttentionTile {
    static_assert((TILE==16 || TILE==32) && (WARPS==4 || WARPS==8));
    static constexpr int QUERY_BYTES=2*8*512*2;
    static constexpr int KEY_BYTES=TILE*512*2;
    static constexpr int PROB_BYTES=2*8*TILE*2;
    static constexpr int SCORE_BYTES=8*TILE*4;
    static constexpr int STATE_BYTES=3*8*4;
    static constexpr int SHARED=QUERY_BYTES+KEY_BYTES+PROB_BYTES+SCORE_BYTES+STATE_BYTES;
    static_assert(SHARED<=64*1024,"P2 tile scratch cap");
};

template<int TILE,int WARPS>
__global__ void attention_tile(const float* query,const uint16_t* raw,const uint16_t* comp,
    const int* selected,const float* sinks,int hist,int p0,int ratio,int T,float* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700 && __CUDA_ARCH__<750
    extern __shared__ uint32_t storage[];
    auto* qhi=reinterpret_cast<uint16_t*>(storage);
    auto* qlo=qhi+8*D;
    auto* keys=qlo+8*D;
    auto* phi=keys+TILE*D;
    auto* plo=phi+8*TILE;
    auto* score=reinterpret_cast<float*>(plo+8*TILE);
    auto* maximum=score+8*TILE;
    auto* denominator=maximum+8;
    auto* rescale=denominator+8;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const int quad=(lane>>2)&3,j=(lane&3)+4*(lane>>4);
    const int t=blockIdx.x,head=blockIdx.y*8,p=p0+t;
    const int nr=min(p+1,128),visible=ratio?(p+1)/ratio:0;
    const int nc=ratio==4?min(512,visible):visible,n=nr+nc;
    const int raw0=p-nr+1-(p0-hist);
    for(int i=tid;i<8*D;i+=WARPS*32)split(query[((size_t)t*H+head)*D+i],qhi[i],qlo[i]);
    if(tid<8){maximum[tid]=sinks[head+tid];denominator[tid]=1.f;}
    // Each warp owns 32 output dimensions per subtile, eight heads. Combine
    // independent PV hi/lo chains only at tile boundaries, then rescale the
    // FP32 running numerator when the online maximum changes.
    float total[16/WARPS][8]={};
    __syncthreads();
    for(int begin=0;begin<n;begin+=TILE){
        for(int i=tid;i<TILE*D;i+=WARPS*32){
            const int row=begin+i/D,d=i%D;
            uint16_t value=0;
            if(row<nr)value=raw[(size_t)(raw0+row)*D+d];
            else if(row<n){const int k=row-nr,id=ratio==4?selected[(size_t)t*512+k]:k;
                value=comp[(size_t)id*D+d];}
            keys[i]=value;
        }
        __syncthreads();
        if(warp==0){
            float hi[8]={},lo[8]={};
            const int key=quad*8;
            for(int k0=0;k0<D;k0+=4){
                uint32_t ah[2]={},al[2]={},b[2]={};
#pragma unroll
                for(int k=0;k<4;++k){
                    const int shift=16*(k%2);
                    ah[k/2]|=uint32_t(qhi[j*D+k0+k])<<shift;
                    al[k/2]|=uint32_t(qlo[j*D+k0+k])<<shift;
                    const uint16_t v=key+j<TILE?keys[(key+j)*D+k0+k]:0;
                    b[k/2]|=uint32_t(v)<<shift;
                }
                strata::kernels::volta::mma884(hi,ah[0],ah[1],b[0],b[1]);
                strata::kernels::volta::mma884(lo,al[0],al[1],b[0],b[1]);
            }
#pragma unroll
            for(int i=0;i<8;++i){
                const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1),c=(lane&2)+4*(i>>2)+(i&1);
                if(key+c<TILE)score[r*TILE+key+c]=(hi[i]+lo[i])*(1.f/sqrtf(float(D)));
            }
        }
        __syncthreads();
        if(tid<8){
            const int count=min(TILE,n-begin);float next=maximum[tid];
            for(int k=0;k<count;++k)next=fmaxf(next,score[tid*TILE+k]);
            const float factor=expf(maximum[tid]-next);rescale[tid]=factor;
            float den=denominator[tid]*factor;
            for(int k=0;k<TILE;++k){
                const float prob=k<count?expf(score[tid*TILE+k]-next):0.f;
                den+=prob;
                split(prob*1024.f,phi[tid*TILE+k],plo[tid*TILE+k]);
            }
            denominator[tid]=den;maximum[tid]=next;
        }
        __syncthreads();
#pragma unroll
        for(int tile=0;tile<16/WARPS;++tile){
            const int dim=(warp+tile*WARPS)*32+quad*8;
            float hi[8]={},lo[8]={};
            for(int k0=0;k0<TILE;k0+=4){
                uint32_t ah[2]={},al[2]={},b[2]={};
#pragma unroll
                for(int k=0;k<4;++k){
                    const int shift=16*(k%2);
                    ah[k/2]|=uint32_t(phi[j*TILE+k0+k])<<shift;
                    al[k/2]|=uint32_t(plo[j*TILE+k0+k])<<shift;
                    b[k/2]|=uint32_t(keys[(k0+k)*D+dim+j])<<shift;
                }
                strata::kernels::volta::mma884(hi,ah[0],ah[1],b[0],b[1]);
                strata::kernels::volta::mma884(lo,al[0],al[1],b[0],b[1]);
            }
#pragma unroll
            for(int i=0;i<8;++i){
                const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1);
                total[tile][i]=total[tile][i]*rescale[r]+(hi[i]+lo[i])/1024.f;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int tile=0;tile<16/WARPS;++tile){
        const int dim=(warp+tile*WARPS)*32+quad*8;
#pragma unroll
        for(int i=0;i<8;++i){
            const int r=(lane&1)+4*(lane>>4)+2*((i>>1)&1),c=(lane&2)+4*(i>>2)+(i&1);
            output[((size_t)t*H+head+r)*D+dim+c]=total[tile][i]/denominator[r];
        }
    }
#endif
}
}
#endif
