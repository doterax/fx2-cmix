// Benchmark: compare original `Lstm` against `LstmFast`.
//
// Both run with the same hyperparameters and are fed identical inputs.
// We measure wall-clock time for a fixed number of Perceive() calls and
// summarize the speedup, plus verify outputs stay close.
//
// Kept short enough to finish comfortably under a minute per implementation
// with the default config. Adjust via the --iters / --steps flags.
//
// Typical usage:
//   ./out/lstm_bench                       # default settings
//   ./out/lstm_bench --iters 20000         # more steps per run
//   perf stat -d ./out/lstm_bench          # perf counters around both runs
//
// The benchmark first runs the baseline `Lstm`, then the new `LstmFast`,
// with identical RNG seeds and inputs, and prints timing + numerical-diff
// statistics.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "mixer/lstm.h"
#include "mixer/lstm_fast.h"
#include "random.hpp"

namespace {

struct Args {
  unsigned int cells        = 200;
  unsigned int layers       = 1;
  unsigned int vocab        = 256;
  unsigned int horizon      = 128;
  unsigned int iters        = 6000;     // number of Perceive() calls per run
  unsigned int warmup       = 64;
  float        learning_rate = 0.03f;
  float        gradient_clip = 10.0f;
  int          bptt_depth   = 0;
  int          bptt_period  = 1;
  unsigned int seed         = 12345;
  int          pattern      = 0;  // 0=random, 1=periodic (learnable)
  int          only         = 0;  // 0=both, 1=baseline only, 2=fast only
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto eq_val = [&](const char* key) -> const char* {
      if (std::strcmp(argv[i], key) == 0 && i + 1 < argc) { return argv[++i]; }
      return nullptr;
    };
    if (const char* v = eq_val("--cells"))   a.cells   = std::atoi(v);
    else if (const char* v = eq_val("--layers")) a.layers = std::atoi(v);
    else if (const char* v = eq_val("--vocab"))  a.vocab  = std::atoi(v);
    else if (const char* v = eq_val("--horizon"))a.horizon= std::atoi(v);
    else if (const char* v = eq_val("--iters"))  a.iters  = std::atoi(v);
    else if (const char* v = eq_val("--warmup")) a.warmup = std::atoi(v);
    else if (const char* v = eq_val("--lr"))     a.learning_rate = std::atof(v);
    else if (const char* v = eq_val("--clip"))   a.gradient_clip = std::atof(v);
    else if (const char* v = eq_val("--seed"))   a.seed   = std::atoi(v);
    else if (const char* v = eq_val("--pattern"))a.pattern= std::atoi(v);
    else if (const char* v = eq_val("--bptt_depth"))  a.bptt_depth  = std::atoi(v);
    else if (const char* v = eq_val("--bptt_period")) a.bptt_period = std::atoi(v);
    else if (const char* v = eq_val("--only")) {
      if      (std::strcmp(v, "baseline") == 0) a.only = 1;
      else if (std::strcmp(v, "fast")     == 0) a.only = 2;
      else                                      a.only = 0;
    }
  }
  return a;
}

// Build deterministic test streams (identical across both runs).
struct TestStream {
  std::vector<unsigned int>     symbols;   // length iters
  std::vector<Eigen::VectorXf>  inputs;    // length iters, each dim=vocab
};

TestStream MakeStream(const Args& a) {
  TestStream s;
  s.symbols.resize(a.iters);
  s.inputs.reserve(a.iters);

  // Simple pseudo-text stream: rotating ASCII-like pattern so that the
  // model has something to actually learn; avoids trivial constant inputs.
  uint32_t x = 0x1234abcd;
  auto rng = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };

  for (unsigned int i = 0; i < a.iters; ++i) {
    unsigned int sym;
    if (a.pattern == 1) {
      // Learnable cyclic pattern: ((i * 7) mod vocab) XOR (i >> 5).
      // Fully deterministic — a competent model should lower NLL well below
      // log(vocab).
      sym = ((i * 7u) ^ (i >> 5)) % a.vocab;
    } else {
      sym = (rng() + i * 31u) % a.vocab;
    }
    s.symbols[i] = sym;
    Eigen::VectorXf v = Eigen::VectorXf::Zero(a.vocab);
    // Soft distribution roughly resembling PPMd output: peak at some symbol.
    unsigned int peak =
        (a.pattern == 1) ? sym : (rng() + i * 7u) % a.vocab;
    float total = 0.0f;
    for (unsigned int j = 0; j < a.vocab; ++j) {
      float d = std::abs(float(int(j) - int(peak)));
      float w = std::exp(-0.02f * d);
      v[j] = w;
      total += w;
    }
    v /= total;
    s.inputs.push_back(std::move(v));
  }
  return s;
}

// Run a model over the stream and accumulate a cross-entropy loss so the
// compiler cannot DCE the work. Returns (seconds, mean_nll, final_probs).
template <typename Model>
double RunModel(Model& m, const TestStream& s, const Args& a,
                double* out_nll, Eigen::VectorXf* final_probs) {
  double       nll   = 0.0;
  unsigned int count = 0;

  // Warmup (excluded from timing).
  for (unsigned int i = 0; i < a.warmup && i < a.iters; ++i) {
    m.SetInput(s.inputs[i]);
    (void)m.Perceive(s.symbols[i]);
  }

  auto t0 = std::chrono::steady_clock::now();
  for (unsigned int i = a.warmup; i < a.iters; ++i) {
    m.SetInput(s.inputs[i]);
    const Eigen::VectorXf& out = m.Perceive(s.symbols[i]);
    unsigned int next = (i + 1 < a.iters) ? s.symbols[i + 1] : s.symbols[i];
    float p = std::max(out[next], 1e-12f);
    nll += -std::log(p);
    ++count;
    if (i + 1 == a.iters && final_probs) *final_probs = out;
  }
  auto t1 = std::chrono::steady_clock::now();

  if (out_nll) *out_nll = (count > 0) ? (nll / count) : 0.0;
  return std::chrono::duration<double>(t1 - t0).count();
}

void PrintHeader(const Args& a) {
  std::printf("LSTM benchmark (Lstm vs LstmFast)\n");
  std::printf("  cells=%u layers=%u vocab=%u horizon=%u iters=%u (warmup=%u)\n",
              a.cells, a.layers, a.vocab, a.horizon, a.iters, a.warmup);
  std::printf("  lr=%.4f clip=%.2f bptt_depth=%d bptt_period=%d seed=%u\n",
              a.learning_rate, a.gradient_clip, a.bptt_depth, a.bptt_period,
              a.seed);
}

}  // namespace

int main(int argc, char** argv) {
  Args a = ParseArgs(argc, argv);

  Eigen::initParallel();
  Eigen::setNbThreads(1);  // single-threaded comparison (per-thread perf)

  PrintHeader(a);

  TestStream s = MakeStream(a);

  Eigen::VectorXf base_final, fast_final;
  double base_nll = 0.0, fast_nll = 0.0;
  double base_s = 0.0, fast_s = 0.0;
  bool   did_base = false, did_fast = false;

  // ---- Baseline Lstm ----
  if (a.only == 0 || a.only == 1) {
    set_seed(a.seed);
    std::unique_ptr<Lstm> base(new Lstm(a.vocab, a.vocab, a.cells, a.layers,
                                        static_cast<int>(a.horizon),
                                        a.learning_rate, a.gradient_clip,
                                        a.bptt_depth, a.bptt_period));
    base_s = RunModel(*base, s, a, &base_nll, &base_final);
    std::printf("\n[baseline  Lstm]      time=%.3fs  mean_nll=%.4f\n",
                base_s, base_nll);
    did_base = true;
  }

  // ---- Fast LstmFast ----
  if (a.only == 0 || a.only == 2) {
    set_seed(a.seed);
    std::unique_ptr<LstmFast> fast(new LstmFast(a.vocab, a.vocab, a.cells, a.layers,
                                            static_cast<int>(a.horizon),
                                            a.learning_rate, a.gradient_clip,
                                            a.bptt_depth, a.bptt_period));
    fast_s = RunModel(*fast, s, a, &fast_nll, &fast_final);
    std::printf("[optimized LstmFast]  time=%.3fs  mean_nll=%.4f\n",
                fast_s, fast_nll);
    did_fast = true;
  }

  // ---- Summary ----
  if (did_base && did_fast) {
    double speedup = base_s / std::max(fast_s, 1e-9);
    std::printf("\nspeedup = %.2fx  (baseline / fast)\n", speedup);
    std::printf("time saved = %.3fs  (%.1f%%)\n", base_s - fast_s,
                100.0 * (base_s - fast_s) / std::max(base_s, 1e-9));
    std::printf("NLL delta = %+.5f  (fast - baseline)\n", fast_nll - base_nll);

    if (base_final.size() == fast_final.size() && base_final.size() > 0) {
      float linf = (base_final - fast_final).cwiseAbs().maxCoeff();
      float l1   = (base_final - fast_final).cwiseAbs().sum();
      std::printf("final_probs: L_inf=%.3e  L1=%.3e (over %ld bins)\n",
                  linf, l1, (long)base_final.size());
    }
  }

  return 0;
}
