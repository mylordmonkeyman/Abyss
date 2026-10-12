// src/core/dsv4_verify.cu - deepseek4's verify window: T (1..kVerifyMax) tokens through the fast path at once, for
// speculative decoding (the engine's n-gram drafts, src/program/generate.cpp).
//
// The layer loop of fast_layers with T columns.  What is shared by the window:
//   * every dense GEMV (attention projections, q_b, wo_a / wo_b, the router and the shared expert's gate/up, the
//     output head) is one gf::mvn launch over the T columns - the weights are read once (DeepSeek V4 Flash's 6.7 GB
//     of them: 4.99 ms for one token, 5.10 for two on an RTX 5090);
//   * the CPU lane's requests: every token's route of a layer is issued before the first answer is awaited, so the
//     host works through the window's RAM-tier experts while the device runs the resident ones.
// What stays per token: the mHC reads, the attention's prep / compress / q_prep / indexer / attend (in position order:
// token t sees exactly positions <= p0 + t - a later token's ring rows and compressed rows are not visible to it), a
// route each, the resident experts' kernels and the CPU-lane combine.
// STRATA_DS4_SPEC_PIPE (default 1): the PIPELINED window instead - each token on its own stream through the
// single-token kernels, token t + 1 a stage behind token t: its attention of a layer runs while the CPU lane computes
// token t's RAM-tier experts of that layer, and token t's next layer runs while the lane does token t + 1's.  The
// dense weights are read once a token (no GEMV sharing), but the host's expert work - the window's largest cost -
// overlaps the device's instead of adding to it.  The order that keeps it exact: a token's attention of a layer
// waits for the previous token's (ring rows, compressor state), and the routes run in (layer, token) order (the
// request counter, the tier tables, the host's in-order answers).
// Rejected tokens are dropped by spec_rewind (the position only): the window ring has kSpecSlack spare rows, and the
// compressor rings / compressed caches are rewritten in position order before anything reads them.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;

namespace {
constexpr size_t kQ8 = 36;   // bytes of a q8_1 block (32 values)

ds::Rope layer_rope_v(const Glm5Geometry& g, bool comp) {
    if (!comp) return ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    return ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx, g.yarn_beta_fast,
                         g.yarn_beta_slow);
}

// the window's buffers: [kVerifyMax] columns of each per-token activation, contiguous (column t at + t * width)
struct VerifyBuf {
    int TM = 0;
    float *R = nullptr;                       // [TM][2][hc * n_embd]: each token's residual pair
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    void* xq = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *iw = nullptr, *q = nullptr, *iq = nullptr;
    void* qr_q = nullptr;
    float *ckv[2] = {nullptr, nullptr}, *csc[2] = {nullptr, nullptr};
    float *o = nullptr, *oa = nullptr;
    void *oq = nullptr, *oaq = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *sh_out = nullptr;
    void* sh_hq = nullptr;
    float *head_x = nullptr, *logits = nullptr, *emb = nullptr;
    void* head_xq = nullptr;
    int* tok = nullptr;                       // device argmax per column
    int* tok_h = nullptr;                     // pinned
    float* emb_h = nullptr;                   // pinned [TM][n_embd]
    std::vector<gf::MoeDev> md;               // a route plan per column (the shared tables, own plan arrays)
    // per column, for the pipelined window's concurrent tokens: the mHC partials, the routed experts' h, the
    // indexer's scores / selection, the fetched experts' scratch slots (column 0: the token path's)
    float* hpart = nullptr;
    unsigned int* hcnt = nullptr;
    void* hq = nullptr;
    float* score = nullptr;
    int* sel = nullptr;
    void* scr = nullptr;
    size_t hpart_n = 0, hq_n = 0, score_n = 0;
    bool pipe = false;                        // the per-column scratch slots allocated: the pipelined window possible
    cudaStream_t st[4] = {};                  // column t's stream (st[0]: the compute stream)
    cudaEvent_t ev_attn[4] = {}, ev_route[4] = {}, ev_fin[4] = {}, ev_start = nullptr;
    cudaEvent_t done = nullptr;
    int T = 0;                                // the last window's size
};
}  // namespace

bool Glm5Model::verify_ready() const {
    const FastState* F = fast_;
    if (F == nullptr || !g_.dsv4() || split_next_ || mtp_il_ >= 0 || F->max_pf > 0 || F->n_ahead > 0) return false;
    for (int il = l0_; il < l1_; ++il)
        if (!F->L[(size_t) il].moe) return false;
    const auto it = ws_map_.find("output.weight");
    return it != ws_map_.end() && it->second.type != 0;
}

bool Glm5Model::verify_alloc(std::string& err) {
    if (vb_ != nullptr) return true;
    const Glm5Geometry& g = g_;
    auto* V = new VerifyBuf();
    const int TM = kVerifyMax;
    V->TM = TM;
    const size_t E = (size_t) g.n_embd, D = (size_t) g.head_dim, HD = (size_t) g.n_head * D,
                 OL = (size_t) g.o_groups * g.o_lora, FF = (size_t) g.n_ff_exp * g.n_shared;
    bool ok = true;
    const auto f = [&](float*& p, size_t n) { ok = ok && cudaMalloc((void**) &p, n * TM * sizeof(float)) == cudaSuccess; };
    const auto b = [&](void*& p, size_t n) { ok = ok && cudaMalloc(&p, n * TM) == cudaSuccess; };
    ok = cudaMalloc((void**) &V->R, (size_t) TM * 2 * g.hc * E * sizeof(float)) == cudaSuccess;
    f(V->x, E); f(V->mixer, E); f(V->ffn, E); f(V->pre, 8); f(V->post, 8); f(V->comb, 16);
    b(V->xq, E / 32 * kQ8);
    f(V->qr_raw, g.q_lora); f(V->qr, g.q_lora); b(V->qr_q, (size_t) g.q_lora / 32 * kQ8);
    f(V->kv_raw, D); f(V->iw, g.idx_heads); f(V->q, HD); f(V->iq, (size_t) g.idx_heads * g.idx_key);
    for (int i = 0; i < 2; ++i) { f(V->ckv[i], 2 * D); f(V->csc[i], 2 * D); }
    f(V->o, HD); b(V->oq, HD / 32 * kQ8); f(V->oa, OL); b(V->oaq, OL / 32 * kQ8);
    f(V->rlog, g.n_expert); f(V->sh_g, FF); f(V->sh_u, FF); f(V->sh_out, E); b(V->sh_hq, FF / 32 * kQ8);
    f(V->head_x, E); b(V->head_xq, E / 32 * kQ8); f(V->logits, (size_t) g.n_vocab); f(V->emb, E);
    ok = ok && cudaMalloc((void**) &V->tok, 64 * sizeof(int)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->tok_h, 64 * sizeof(int), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->emb_h, (size_t) TM * E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaEventCreateWithFlags(&V->done, cudaEventDisableTiming) == cudaSuccess;
    V->hpart_n = (size_t) (4 * E / 512) * 26 + 64;
    V->hq_n = (size_t) 8 * g.n_ff_exp / 32 * kQ8;
    V->score_n = (size_t) std::max<int64_t>(64, max_ctx_ / 4 + 64);
    ok = ok && cudaMalloc((void**) &V->hpart, V->hpart_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->hcnt, (size_t) 64 * TM) == cudaSuccess;
    ok = ok && cudaMalloc(&V->hq, V->hq_n * TM) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->score, V->score_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->sel, (size_t) g.idx_top_k * TM * sizeof(int)) == cudaSuccess;
    if (ok) cudaMemset(V->hcnt, 0, (size_t) 64 * TM);
    V->st[0] = fast_->cs;
    for (int t = 0; t < TM && ok; ++t) {
        if (t > 0) ok = cudaStreamCreateWithFlags(&V->st[t], cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* e : {&V->ev_attn[t], &V->ev_route[t], &V->ev_fin[t]})
            ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
    }
    ok = ok && cudaEventCreateWithFlags(&V->ev_start, cudaEventDisableTiming) == cudaSuccess;
    // the per-column route plans: everything the route / wait / fetch / gate_up / down / cpu_wait kernels write per
    // route; the tables, the request counter, the rings, the scratch slots and the route counts stay shared
    V->md.assign((size_t) TM, fast_->md);
    for (int t = 0; t < TM && ok; ++t) {
        gf::MoeDev& m = V->md[(size_t) t];
        uint8_t* blk = nullptr;
        const size_t nb = 4 * 8 * sizeof(unsigned long long) + 3 * 64 + E * sizeof(float);
        ok = cudaMalloc((void**) &blk, nb) == cudaSuccess;
        if (!ok) break;
        cudaMemset(blk, 0, nb);
        m.plan_ptr = (unsigned long long*) blk;
        m.fetch_src = (unsigned long long*) (blk + 64);
        m.plan_w = (float*) (blk + 128);
        m.plan_id = (int*) (blk + 192);
        m.wait_seq = (unsigned int*) (blk + 256);
        m.cpu_flag = (int*) (blk + 320);
        m.cpu_part = (float*) (blk + 448);
        if (fast_->md.cpu_seq != nullptr) m.cpu_seq = (unsigned int*) (blk + 384);
        m.pf_src = fast_->pf_buf[0];
        m.pf_dst = fast_->pf_buf[0] + 8;
        m.pf_n = fast_->pf_n_buf[0];
    }
    // the pipelined window's scratch slots (8 an extra column, an expert blob each): without them, the batched window
    size_t blob_max = 0;
    for (int il = l0_; il < l1_; ++il) blob_max = std::max(blob_max, fast_->L[(size_t) il].blob);
    const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
    if (ok && sstride > 0 && cudaMalloc(&V->scr, (size_t) (TM - 1) * 8 * sstride) == cudaSuccess) {
        for (int t = 1; t < TM && ok; ++t) {
            unsigned long long sp[8];
            for (int i = 0; i < 8; ++i)
                sp[i] = (unsigned long long) ((uint8_t*) V->scr + ((size_t) (t - 1) * 8 + i) * sstride);
            unsigned long long* d = nullptr;
            ok = cudaMalloc((void**) &d, sizeof sp) == cudaSuccess &&
                 cudaMemcpy(d, sp, sizeof sp, cudaMemcpyHostToDevice) == cudaSuccess;
            V->md[(size_t) t].scratch = d;
        }
        V->pipe = ok;
    } else {
        cudaGetLastError();
        std::fprintf(stderr, "deepseek4 verify: no VRAM for the pipelined window's scratch slots (%.0f MB) - batched\n",
                     (double) (TM - 1) * 8 * sstride / 1048576.0);
    }
    if (!ok) {
        cudaGetLastError();
        err = "deepseek4 verify: the window's buffers did not allocate";
        delete V;
        return false;
    }
    vb_ = V;
    return true;
}

bool Glm5Model::verify_attn(int il, int64_t p0, int tb, int T, cudaStream_t s, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int D = g.head_dim, r = Ly.ratio, E = g.n_embd;
    const ds::Rope rope = layer_rope_v(g, r != 0), crope = layer_rope_v(g, true);
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora;
    // the columns tb.. (the window's buffers from column tb on)
    const size_t q8e = (size_t) (E / 32) * kQ8;
    void* const xq = (uint8_t*) V->xq + tb * q8e;
    float* const x = V->x + (size_t) tb * E;
    float* const qr_raw = V->qr_raw + (size_t) tb * g.q_lora;
    float* const qr = V->qr + (size_t) tb * g.q_lora;
    void* const qr_qb = (uint8_t*) V->qr_q + (size_t) tb * (g.q_lora / 32) * kQ8;
    float* const kv_raw = V->kv_raw + (size_t) tb * D;
    float* const iwb = V->iw + (size_t) tb * g.idx_heads;
    float* const ckv0 = V->ckv[0] + (size_t) tb * 2 * D;
    float* const csc0 = V->csc[0] + (size_t) tb * 2 * D;
    float* const ckv1 = V->ckv[1] + (size_t) tb * 2 * D;
    float* const csc1 = V->csc[1] + (size_t) tb * 2 * D;
    float* const qb = V->q + (size_t) tb * HD;
    float* const iqb = V->iq + (size_t) tb * g.idx_heads * g.idx_key;
    float* const ob = V->o + (size_t) tb * HD;
    void* const oqb = (uint8_t*) V->oq + (size_t) tb * (HD / 32) * kQ8;
    float* const oab = V->oa + (size_t) tb * OL;
    void* const oaqb = (uint8_t*) V->oaq + (size_t) tb * (OL / 32) * kQ8;
    float* const mixer = V->mixer + (size_t) tb * E;
    // 1. the projections of x, every column at once
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    const int cw = r == 4 ? 2 * D : D;
    j[nj++] = {Ly.q_a.q, xq, x, qr_raw, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
    j[nj++] = {Ly.kv.q, xq, x, kv_raw, nullptr, 1.0f, Ly.kv.type, E, D};
    if (r != 0) {
        j[nj++] = {Ly.comp_kv.q, xq, x, ckv0, nullptr, 1.0f, Ly.comp_kv.type, E, cw, 0, 2 * D};
        j[nj++] = {Ly.comp_gate.q, xq, x, csc0, nullptr, 1.0f, Ly.comp_gate.type, E, cw, 0, 2 * D};
    }
    if (r == 4) {
        j[nj++] = {Ly.idx_comp_kv.q, xq, x, ckv1, nullptr, 1.0f, Ly.idx_comp_kv.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_comp_gate.q, xq, x, csc1, nullptr, 1.0f, Ly.idx_comp_gate.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_proj_f, nullptr, x, iwb, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, E, g.idx_heads};
    }
    if (!gf::mvn(j, nj, T, s)) { err = "deepseek4 verify: attention projections"; return false; }
    // 2. per token, in position order: norms, RoPE, the ring writes, a block end's compression
    for (int t = 0; t < T; ++t) {
        const int p = (int) (p0 + tb + t);
        ds::PrepArgs a;
        a.qr_raw = qr_raw + (size_t) t * g.q_lora;
        a.q_a_norm = Ly.q_a_norm;
        a.qr = qr + (size_t) t * g.q_lora;
        a.qr_q = (uint8_t*) qr_qb + (size_t) t * (g.q_lora / 32) * kQ8;
        a.q_lora = g.q_lora;
        a.kv_raw = kv_raw + (size_t) t * D;
        a.kv_norm = Ly.kv_a_norm;
        a.raw = Ly.raw;
        a.ring = ds_ring_;
        a.head_dim = D;
        a.n_rot = g.n_rot;
        a.rope = rope;
        a.comp_rope = crope;
        a.p = p;
        a.eps = g.norm_eps;
        if (r != 0) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.ring_kv;
            c.ring_sc = Ly.ring_sc;
            c.cache = Ly.comp_cache;
            c.ape = Ly.comp_ape;
            c.norm = Ly.comp_norm;
            c.D = D;
            c.ratio = r;
            c.overlap = r == 4;
            a.comp_kv[a.n_comp] = ckv0 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc0 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        if (r == 4) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.iring_kv;
            c.ring_sc = Ly.iring_sc;
            c.cache = Ly.idx_cache;
            c.ape = Ly.idx_comp_ape;
            c.norm = Ly.idx_comp_norm;
            c.D = g.idx_key;
            c.ratio = 4;
            c.overlap = 1;
            a.comp_kv[a.n_comp] = ckv1 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc1 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        ds::prep(a, s);
        if (r != 0 && (p + 1) % r == 0) ds::compress(a, s);
    }
    // 3. q_b (and the indexer's q), every column at once
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, qr_qb, qr, qb, nullptr, 1.0f, Ly.q_b.type, g.q_lora, HD};
    if (r == 4)
        j2[nj2++] = {Ly.idx_q_b_q.q, qr_qb, qr, iqb, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mvn(j2, nj2, T, s)) { err = "deepseek4 verify: q_b"; return false; }
    // 4. per token: the per-head norm + RoPE, the indexer's top-k, the attention
    for (int t = 0; t < T; ++t) {
        const int c = tb + t, p = (int) (p0 + c);
        float* q = qb + (size_t) t * HD;
        float* iq = iqb + (size_t) t * g.idx_heads * g.idx_key;
        ds::q_prep(q, g.n_head, D, g.n_rot, rope, r == 4 ? iq : nullptr, g.idx_heads, g.idx_key, crope, p, g.norm_eps,
                   s);
        const int n_raw = std::min(p + 1, g.n_swa);
        int n_comp = 0;
        const int* sel = nullptr;
        if (r != 0) {
            const int n_vis = (p + 1) / r;
            if (r == 4 && n_vis > 0) {
                int* cs = V->sel + (size_t) c * g.idx_top_k;
                ds::indexer(iq, Ly.idx_cache, iwb + (size_t) t * g.idx_heads, g.idx_heads, g.idx_key, n_vis,
                            g.idx_top_k, V->score + (size_t) c * V->score_n, cs, s);
                n_comp = std::min(n_vis, g.idx_top_k);
                sel = cs;
            } else {
                n_comp = n_vis;
            }
        }
        ds::attend(q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
                   ob + (size_t) t * HD, s, c);
    }
    // 5. the grouped output LoRA, every column at once (a group's rows of column t at o + t * HD + gi * gin)
    gf::quantize_q8_1(ob, oqb, T * HD, s);
    const int gin = g.n_head / g.o_groups * D;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek4 verify: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) oqb + (size_t) gi * (gin / 32) * kQ8, ob + (size_t) gi * gin,
                  oab + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora, HD / 32, OL};
    if (!gf::mvn(jo, g.o_groups, T, s)) { err = "deepseek4 verify: wo_a"; return false; }
    gf::quantize_q8_1(oab, oaqb, T * OL, s);
    gf::MvJob jb = {Ly.out_b.q, oaqb, oab, mixer, nullptr, 1.0f, Ly.out_b.type, OL, E};
    if (!gf::mvn(&jb, 1, T, s)) { err = "deepseek4 verify: wo_b"; return false; }
    return true;
}

bool Glm5Model::verify_moe(int il, int tb, int T, const std::vector<int32_t>& toks, cudaStream_t s,
                           cudaEvent_t wait_ev, cudaEvent_t done_ev, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared, k = g.n_exp_used;
    const size_t q8e = (size_t) (E / 32) * kQ8;
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, V->x + (size_t) tb * E, V->rlog + (size_t) tb * g.n_expert, nullptr, 1.0f,
            gf::kTypeBF16, E, g.n_expert};
    j[1] = {Ly.sh_gate.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_g + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_u + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_up.type, E, FFs};
    if (!gf::mvn(j, 3, T, s)) { err = "deepseek4 verify: router / shared expert"; return false; }
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    static const int pf_rank = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_RANK");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    // every token's route first: the CPU lane has the whole window's RAM-tier experts queued before any wait
    if (wait_ev != nullptr) cudaStreamWaitEvent(s, wait_ev, 0);
    for (int t = tb; t < tb + T; ++t) {
        gf::moe_route(V->rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, k, g.w_scale, g.norm_w != 0, il,
                      V->x + (size_t) t * E, E, V->md[(size_t) t], V->sh_g + (size_t) t * FFs,
                      V->sh_u + (size_t) t * FFs, g.swiglu_shexp, FFs, (uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                      s, nullptr, nullptr, 0, nullptr, nullptr, 0, fast_skip_from(il), lane ? F->cpu_plan : 0ull, promote_min, pf_rank,
                      g.gating, Ly.tid2eid != nullptr ? Ly.tid2eid + (size_t) toks[(size_t) t] * k : nullptr,
                      fast_skip_w(il));
        ++F->expected;
    }
    // ... then each token's resident experts (its own plan and h) ...
    for (int t = tb; t < tb + T; ++t) {
        const gf::MoeDev& md = V->md[(size_t) t];
        float* sh_out = V->sh_out + (size_t) t * E;
        void* hq = (uint8_t*) V->hq + (size_t) t * V->hq_n;
        gf::moe_wait(md, E, s);
        gf::moe_fetch(md, k, Ly.blob, s);
        // the next route may find this one's promotions resident: it goes after their copies have landed
        if (done_ev != nullptr && t == tb + T - 1) cudaEventRecord(done_ev, s);
        gf::moe_gate_up(Ly.gu_type, md, k, E, g.n_ff_exp, g.swiglu_exp, (const uint8_t*) V->xq + (size_t) t * q8e,
                        hq, Ly.sh_down.q, Ly.sh_down.type, (const uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                        FFs, sh_out, s);
        gf::moe_down(Ly.d_type, md, k, E, g.n_ff_exp, Ly.down_off, hq, Ly.sh_down.q != nullptr ? sh_out : nullptr,
                     V->ffn + (size_t) t * E, s);
    }
    // ... and the CPU lane's answers, in request order
    if (lane)
        for (int t = tb; t < tb + T; ++t) gf::moe_cpu_wait(V->md[(size_t) t], E, V->ffn + (size_t) t * E, s);
    return true;
}

bool Glm5Model::verify(const std::vector<int32_t>& toks, std::string& err) {
    const int T = (int) toks.size();
    if (T < 1 || T > kVerifyMax) {
        err = "deepseek4 verify: 1.." + std::to_string(kVerifyMax) + " tokens";
        return false;
    }
    if (!verify_ready()) {
        err = "deepseek4 verify: not available (deepseek4 fast path on one GPU, no MTP / prefetch / lookahead)";
        return false;
    }
    if (pos_ + T > max_ctx_) {
        err = "deepseek4 verify: the window exceeds max_ctx";
        return false;
    }
    if (!verify_alloc(err)) return false;
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    const int64_t p0 = pos_;
    pos_ += T;
    V->T = T;
    if (!fast_boundary(err)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    // ---- the embedding rows -> each token's 4 streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "deepseek4 verify: no embedding dequantizer";
        return false;
    }
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E;
    for (int t = 0; t < T; ++t)
        tt->to_float(pack_emb_src_ + (size_t) toks[(size_t) t] * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd),
                     V->emb_h + (size_t) t * E, g.n_embd);
    cudaMemcpyAsync(V->emb, V->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, s);
    std::vector<float*> Rc((size_t) T), Ro((size_t) T);
    for (int t = 0; t < T; ++t) {
        Rc[(size_t) t] = V->R + (size_t) t * 2 * HCE;
        Ro[(size_t) t] = Rc[(size_t) t] + HCE;
        gf::embed_streams(V->emb + (size_t) t * E, Rc[(size_t) t], g.n_embd, s);
    }
    const auto hc_read = [&](int il, int t, const float* block_out, bool attn, cudaStream_t hs) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[(size_t) t];
        h.R_new = Ro[(size_t) t];
        h.post_in = V->post + (size_t) t * 8;
        h.comb_in = V->comb + (size_t) t * 16;
        h.pre = V->pre + (size_t) t * 8;
        h.post = V->post + (size_t) t * 8;
        h.comb = V->comb + (size_t) t * 16;
        h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
        h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
        h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
        h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = V->x + (size_t) t * E;
        h.xq = (uint8_t*) V->xq + (size_t) t * (E / 32) * kQ8;
        h.part = V->hpart + (size_t) t * V->hpart_n;
        h.counter = V->hcnt + (size_t) t * 16;
        gf::hc(h, hs);
        if (block_out != nullptr) std::swap(Rc[(size_t) t], Ro[(size_t) t]);
    };
    // STRATA_GLM_PROF (as the token path's): the window's phases, marked v<T>_* so the two kinds stay apart
    // the pipelined window (STRATA_DS4_SPEC_PIPE, default on; not with PROF - its marks are one stream's)
    static const bool pipe_env = [] {
        const char* v = getenv("STRATA_DS4_SPEC_PIPE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const bool pipe_env1 = getenv("STRATA_DS4_SPEC_PIPE") != nullptr && std::atoi(getenv("STRATA_DS4_SPEC_PIPE")) == 2;
    const bool pipe = pipe_env && V->pipe && T > 1 && !F->prof_on;
    const bool prof = F->prof_on;
    const std::string vt = "v" + std::to_string(T) + "_";
    static std::vector<std::string> names;   // (mark keeps the pointer)
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"hc_attn", "attn", "hc_ffn", "moe"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 4 + ph].c_str()); };
    if (prof) F->mark("v_start");
    if (pipe) {
        // column t on stream t, after the embedding; per layer, its attention after column t - 1's (ring rows,
        // compressor state) and its route after column t - 1's (column 0: after the last column's of the layer before)
        cudaEventRecord(V->ev_start, s);
        for (int t = 1; t < T; ++t) cudaStreamWaitEvent(V->st[t], V->ev_start, 0);
        for (int il = l0_; il < l1_; ++il)
            for (int t = 0; t < T; ++t) {
                cudaStream_t st = pipe_env1 ? s : V->st[t];
                if (t > 0) cudaStreamWaitEvent(st, V->ev_attn[t - 1], 0);
                hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, st);
                if (!verify_attn(il, p0, t, 1, st, err)) return false;
                cudaEventRecord(V->ev_attn[t], st);
                hc_read(il, t, V->mixer + (size_t) t * E, false, st);
                cudaEvent_t after = t > 0 ? V->ev_route[t - 1] : il > l0_ ? V->ev_route[T - 1] : nullptr;
                if (!verify_moe(il, t, 1, toks, st, after, V->ev_route[t], err)) return false;
            }
    } else {
        for (int il = l0_; il < l1_; ++il) {
            for (int t = 0; t < T; ++t) hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, s);
            mark(0);
            if (!verify_attn(il, p0, 0, T, s, err)) return false;
            mark(1);
            for (int t = 0; t < T; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false, s);
            mark(2);
            if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
            mark(3);
        }
    }
    // the last layer's write half, the head, the output rows (every column at once) and each column's argmax
    const auto fn = w16_.find("output_hc_fn.weight");
    const auto sc = w_.find("output_hc_scale.weight");
    const auto bs = w_.find("output_hc_base.weight");
    const auto nw = w_.find("output_norm.weight");
    if (fn == w16_.end() || sc == w_.end() || bs == w_.end() || nw == w_.end()) {
        err = "deepseek4 verify: the head's tensors";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        cudaStream_t st = pipe && !pipe_env1 ? V->st[t] : s;
        gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                    g.n_embd, Ro[(size_t) t], st);
        ds::head_prep(Ro[(size_t) t], fn->second, sc->second, bs->second, g.hc_eps, nw->second, g.norm_eps, g.n_embd,
                      V->head_x + (size_t) t * E, (uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8, st);
        if (pipe && t > 0) {
            cudaEventRecord(V->ev_fin[t], st);
            cudaStreamWaitEvent(s, V->ev_fin[t], 0);
        }
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, T, s)) { err = "deepseek4 verify: output head"; return false; }
    } else {
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "deepseek4 verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
    cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(V->done, s);
    cudaError_t done;
    while ((done = cudaEventQuery(V->done)) == cudaErrorNotReady) std::this_thread::yield();
    if (!glmfast::cuda_ok(done, "deepseek4 verify completion", err) || !F->route_ok(err)) return false;
    if (gf::launch_errors() > 0) {
        err = "deepseek4 verify: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    F->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (prof) F->collect();
    last_tok_ = V->tok_h[T - 1];
    return true;
}

int Glm5Model::verify_argmax(int i) const {
    const auto* V = (const VerifyBuf*) vb_;
    return V != nullptr && i >= 0 && i < V->T ? V->tok_h[i] : -1;
}

bool Glm5Model::verify_logits(int i, std::vector<float>& out, std::string& err) {
    auto* V = (VerifyBuf*) vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return false;
    }
    cudaSetDevice(dev_);
    out.resize((size_t) g_.n_vocab);
    if (cudaMemcpy(out.data(), V->logits + (size_t) i * g_.n_vocab, out.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "deepseek4 verify: logits copy";
        return false;
    }
    return true;
}

int Glm5Model::verify_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) {
    auto* V = (VerifyBuf*) vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return -1;
    }
    if (sp.greedy) return V->tok_h[i];
    cudaSetDevice(dev_);
    cudaMemcpyAsync(sc_ + sc_logits, V->logits + (size_t) i * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float),
                    cudaMemcpyDeviceToDevice, fast_->cs);
    return fast_sample(sp, err);
}

}  // namespace strata::core
