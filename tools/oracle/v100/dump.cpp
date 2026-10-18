// Bounded, CPU-only teacher-forced oracle against the pinned upstream llama.cpp.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: cpu_oracle model.gguf tokens.txt logits.f32\n");
        return 2;
    }
    std::ifstream input(argv[2]);
    if (!input) return 2;
    std::string text((std::istreambuf_iterator<char>(input)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    std::vector<llama_token> ids;
    int64_t value;
    while (stream >> value) {
        if (value < 0 || value > std::numeric_limits<int32_t>::max() || ids.size() >= 256) return 2;
        ids.push_back(static_cast<llama_token>(value));
    }
    if (!stream.eof() || ids.size() < 2) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = true;
    mp.use_mlock = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) return 1;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (auto id : ids) if (id >= vocab) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 32;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.no_perf = true;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) return 1;
    std::unique_ptr<std::FILE, decltype(&std::fclose)> out(std::fopen(argv[3], "wb"), std::fclose);
    if (!out) return 1;
    // Only positions predicting supplied tokens: the final input needs no forward.
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
        auto one = ids[i];
        auto batch = llama_batch_get_one(&one, 1);
        if (llama_decode(ctx.get(), batch)) return 1;
        const float * logits = llama_get_logits_ith(ctx.get(), 0);
        if (!logits) return 1;
        int best = 0;
        for (int j = 0; j < vocab; ++j) {
            if (!std::isfinite(logits[j])) return 1;
            if (logits[j] > logits[best]) best = j;
        }
        if (std::fwrite(logits, sizeof(float), vocab, out.get()) != static_cast<size_t>(vocab)) return 1;
        std::printf("position %zu argmax %d\n", i, best);
        std::fflush(stdout);
    }
    std::printf("WROTE %zu %d\n", ids.size() - 1, vocab);
    return 0;
}
