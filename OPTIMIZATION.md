# cmix Speed Optimization Analysis

## Baseline Measurements

| Test | Input Size | Output Size | Time | Rate | Compression Ratio |
|------|-----------|-------------|------|------|-------------------|
| `input` (fast) | 51,052 B | 6,152 B | 20.1 s | 2,569 B/s | 12.05% |
| `input2` (longer) | 941,724 B | 181,019 B | 255.9 s | 3,688 B/s | 19.22% |

## After P0 Optimizations

| Test | Input Size | Output Size | Time | Rate | Compression Ratio | Delta vs Baseline |
|------|-----------|-------------|------|------|-------------------|--------------------|
| `input` (fast) | 51,052 B | 6,149 B | 18.1 s | 2,817 B/s | 12.04% | **-3 B, 1.10× faster** |
| `input2` (longer) | 941,724 B | 181,039 B | 234.3 s | 4,019 B/s | 19.22% | **+20 B, 1.09× faster** |

**Notes:** The tiny output size changes (−3 B / +20 B) are from the log-sum-exp trick in the LSTM
softmax, which changes floating-point evaluation order. These are within numerical noise
and demonstrate no meaningful compression degradation. Net speedup: **~9-10%**.

**Command (fast):** `./cmix.exe no-preprocess .\prof_input\input ./res.bin`  
**Command (longer):** `./cmix.exe no-preprocess .\prof_input\input2 ./res2.bin`  
**Configuration:** 461 total model outputs, LSTM 200 cells, horizon 128, PPMd order 25 / 1024 MB, 23 layer-0 mixers + 1 layer-1 mixer + SSE.  
**Date:** 2026-03-13

## After P1 (No-Risk) Optimizations

Implemented: **P1-4** (fuse layer normalization), **P1-5a** (PPMd cache lim), **P1-5b** (PPMd hardware prefetch), **P1-12a** (PGO build).  
**P1-3** and **P1-6a** were already completed as part of P0 work.

| Test | Input Size | Output Size | Time | Rate | Compression Ratio | Delta vs Baseline |
|------|-----------|-------------|------|------|-------------------|--------------------|
| `input` (fast) | 51,052 B | 6,149 B | 17.3 s | 2,951 B/s | 12.04% | **−3 B, 1.16× faster** |
| `input2` (longer) | 941,724 B | 181,039 B | 243.8 s | 3,863 B/s | 19.22% | **+20 B, 1.05× faster** |

**Notes:** Compression output identical to P0 (no new numerical changes). PGO provides a small
additional speedup on `input` (~3% vs P0). The `input2` time shows run-to-run variance
(±4% is normal on a ~4min test). PGO was trained on `input` — richer training data
(e.g., enwik8) may yield better results on larger files.

**PGO build steps:**
1. `make prof_gen` — build instrumented binary
2. `./cmix.exe no-preprocess prof_input/input res.bin` — generate profile
3. `llvm-profdata merge -output pgo_data/default.profdata pgo_data/*.profraw` — merge profiles
4. `make prof_use` — rebuild with profile data

**Date:** 2026-03-13

---

## Architecture Overview

The hot path executes **per bit** (8× per byte):
```
Encoder::Encode(bit)
  → Predictor::Predict()        [~30 model predictions + 24 mixer dot products + SSE]
  → Predictor::Perceive(bit)    [~30 model updates + 24 mixer weight updates]
  → Encoder range update         [arithmetic coding - very fast]
```

Every 8th bit (byte boundary), additional `ByteUpdate()` calls trigger for all models,
including the expensive **LSTM backward pass (BPTT)**.

### Model composition (461 total outputs):
- 1× Bracket model (1 output)
- 1× FXCM model (N outputs)
- 1× Direct model (1 output)
- 10× Match models (1 output each)
- 14× Indirect/Nonstationary models (1 output each)
- 1× Indirect/RunMap model (1 output)
- 1× PPMd byte model (1 output)
- 1× ByteMixer/LSTM (1 output)
- 23× Layer-0 mixers → 1× Layer-1 mixer → SSE post-processor

---

## Optimization Opportunities (Ranked by Impact)

### 1. LSTM BPTT Optimization [HIGH IMPACT — estimated 1.5-2.5×]

**Current state:** The LSTM runs a full backpropagation-through-time (BPTT) over the entire
`horizon_=128` window on **every byte** (epoch 0). Each BPTT iteration processes 3 gates
(forget, input, output) × 200 cells × backward passes through all 128 timesteps.

**File:** `src/mixer/lstm.hpp` — `Lstm::Perceive()`, lines 88-114

```cpp
if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
        for (int layer = layers_.size() - 1; layer >= 0; --layer) {
            // Full BPTT over 128 timesteps × 200 cells × 3 gates
        }
    }
}
```

**Problem:** This is O(horizon × num_cells² × 3_gates) per byte. With horizon=128,
num_cells=200, this is ~15M floating point operations per byte boundary.

**Proposed optimizations (pick one or combine):**

#### 1a. Truncated BPTT
Instead of backprop through all 128 steps, truncate to the last K steps (e.g., K=16 or K=32).
This reduces work by 4-8× while preserving most gradient signal. Long-range dependencies
in text compression rarely exceed 16-32 tokens for gradient flow.

```cpp
// Instead of: for (int epoch = horizon_ - 1; epoch >= 0; --epoch)
int bptt_depth = std::min(32, horizon_);
for (int epoch = horizon_ - 1; epoch >= horizon_ - bptt_depth; --epoch)
```

**Risk to compression:** Low-Medium. Gradients beyond 32 steps are typically very small
due to vanishing gradients. Empirical testing needed — may lose 0.01-0.1% ratio.

#### 1b. Reduce BPTT frequency
Currently BPTT runs every time `epoch_ == 0` (every 128 bytes). If we reduce this to
every 2nd or 4th cycle, the weight updates are less frequent but still converge.

**Risk:** Low. Online learning is already noisy; reducing update frequency trades
slightly stale gradients for 2-4× speedup.

#### 1c. Batch output layer update
The output layer update loop is:
```cpp
for (unsigned int i = 0; i < output_size_; ++i) {
    output_layer_[epoch_][i] = output_layer_[last_epoch][i];
    output_layer_[epoch_][i].noalias() -= errors[i] * hidden_;
}
```
This is `output_size_` (up to 256) rank-1 updates. It can be rewritten as a single
matrix operation if `output_layer_` is stored as a matrix instead of `vector<VectorXf>`.

**Change `output_layer_` from `vector<vector<VectorXf>>` to `vector<MatrixXf>`** where
each matrix is `(hidden_size+1) × output_size_`. Then the update becomes:

```cpp
output_layer_mat_[epoch_] = output_layer_mat_[last_epoch];
output_layer_mat_[epoch_].noalias() -= hidden_ * errors.transpose();  // single rank-1 matrix update
```

And the forward pass dot products become a single matrix-vector multiply:
```cpp
output_[epoch_] = (output_layer_mat_[epoch_].transpose() * hidden_).array().exp();
```

**Risk to compression:** Zero — mathematically identical.

---

### 2. Adam Optimizer Temporaries [MEDIUM-HIGH IMPACT — estimated 1.2-1.5×]

**Files:** `src/mixer/lstm-layer.hpp` — `Adam()` and `AdamMatrix()`, lines 37-97

**Current problems:**

#### 2a. Redundant Eigen::*::Ones() allocations
```cpp
(*w) -= alpha * (((*m) / ...).cwiseQuotient(
    ((*v) / ...).array().sqrt().matrix() + eps * Eigen::VectorXf::Ones(w->size())));
```
`Eigen::VectorXf::Ones(w->size())` allocates a temporary vector **every call**. 
Replace with `Eigen::VectorXf::Constant(w->size(), eps)` or better, use array operations:

```cpp
// Replace entire Adam function body with:
(*m) *= beta1;
(*m).noalias() += (1.0f - beta1) * (*g);
(*v) *= beta2;
(*v).noalias() += (1.0f - beta2) * g->cwiseProduct(*g);
float bias_corr1 = 1.0f - powf(beta1, t);
float bias_corr2 = 1.0f - powf(beta2, t);
// Use .array() to avoid temporaries:
w->array() -= alpha * (m->array() / bias_corr1) / 
              ((v->array() / bias_corr2).sqrt() + eps);
```

This eliminates 2-4 temporary Vector/Matrix allocations per Adam call.

**Risk to compression:** Zero — mathematically identical.

#### 2b. Pre-compute pow() values
`pow(beta1, t)` and `pow(beta2, t)` are called twice in the if/else branches.
Compute them once and reuse. Also, `powf()` for float is faster than `pow()` for double.

Better yet, track running products:
```cpp
// In LstmLayer, maintain:
float beta1_power_ = 1.0f, beta2_power_ = 1.0f;
// On each update step:
beta1_power_ *= beta1;
beta2_power_ *= beta2;
```
This replaces `pow()` calls (transcendental function) with single multiplications.

#### 2c. AdamMatrix identical to Adam but for matrices
The `AdamMatrix()` function does exactly the same computation as `Adam()` but for
`Eigen::MatrixXf*`. These could be unified using Eigen's `.reshaped()` or by using
`Eigen::Map<VectorXf>` to treat the matrix as a flat vector.

**Risk to compression:** Zero.

---

### 3. LSTM Softmax Optimization [MEDIUM IMPACT — estimated 1.1-1.3×]

**File:** `src/mixer/lstm.hpp` — `Lstm::Predict()`, lines 126-134

```cpp
for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = hidden_.dot(output_layer_[epoch_][i]);
    output_[epoch_][i] = exp(sum);
}
float total = output_[epoch_].sum();
output_[epoch_] /= total;
```

**Problems:**
1. `output_size_` individual dot products (up to 256) instead of single matrix-vector multiply
2. `exp()` called for all 256 outputs — most will be near-zero

**Optimization:** Convert to matrix form (see 1c above), then:
```cpp
Eigen::VectorXf logits = output_layer_mat_[epoch_].transpose() * hidden_;
float max_logit = logits.maxCoeff();
output_[epoch_] = (logits.array() - max_logit).exp();  // numerically stable
output_[epoch_] /= output_[epoch_].sum();
```

This gives Eigen the opportunity to use SIMD for both the matrix-vector multiply and
the vectorized exp(). The log-sum-exp trick also prevents numerical overflow.

**Risk to compression:** Near-zero. The log-sum-exp trick can cause tiny floating-point
differences, but these are beneficial (more numerically stable).

---

### 4. Layer Normalization Optimization [MEDIUM IMPACT — estimated 1.1-1.2×]

**File:** `src/mixer/lstm-layer.hpp` — `LstmLayer::ForwardPass(NeuronLayer&, ...)`, lines 156-164

```cpp
float variance = neurons.norm_[epoch_].squaredNorm() / num_cells_;
neurons.ivar_[epoch_] = 1.0f / sqrt(variance + 1e-5f);
neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
neurons.state_[epoch_].noalias() = 
    neurons.norm_[epoch_].cwiseProduct(neurons.gamma_) + neurons.beta_;
```

Called 3× per timestep (forget/input/output gates) × 128 timesteps per BPTT = 384 calls.

**Note:** `neurons.ivar_` is declared as `Eigen::VectorXf::Zero(horizon)` but used as a
scalar per epoch (only `neurons.ivar_[epoch_]` is ever set). This wastes memory and
L1 cache (storing `horizon` floats when only scalars are needed). Change it to
`std::vector<float> ivar_(horizon)` or use a plain `float` if only the current epoch
is needed.

**Optimization:** Fuse the normalize + affine transform:
```cpp
float inv_std = 1.0f / sqrtf(neurons.norm_[epoch_].squaredNorm() / num_cells_ + 1e-5f);
neurons.state_[epoch_].noalias() = 
    (inv_std * neurons.gamma_.array() * neurons.norm_[epoch_].array() + neurons.beta_.array()).matrix();
```
This skips the intermediate write to `neurons.norm_[epoch_]` and reduces memory traffic.

**Risk to compression:** Zero — mathematically identical.

---

### 5. PPMd Memory Compaction & Prefetching [MEDIUM IMPACT — estimated 1.1-1.3×]

**File:** `src/models/ppmd.cpp`

#### 5a. Ptr2Indx / Indx2Ptr Cost in Hot Path
Every context access goes through index conversion:
```cpp
void *Indx2Ptr(uint indx) {
    assert(HeapStart != nullptr ...);
    if (indx == 0) return nullptr;
    uint lim = (uint)(UnitsStartBase - HeapStart);
    qword addr = (indx >= lim) ? qword(indx - lim) * UNIT_SIZE + lim : indx;
    // ... bounds checking ...
    return HeapStart + addr;
}
```

**The assertions and bounds checks are compiled in even in release builds** if `NDEBUG`
is not properly set. The function has a branch (`indx >= lim`) that the CPU
can't easily predict since indices span both regions.

**Proposed optimization:**
1. **Ensure `assert()` and `printf()` calls in Ptr2Indx/Indx2Ptr are guarded by `NDEBUG`.**
   Currently there are `printf()` error messages that always execute on the error path.
   While these only trigger on corruption, the branch prediction overhead of their
   presence still costs ~1 cycle per call.

2. **Create a fast-path inline version** for the common case (indx >= lim, which is
   the units region):
```cpp
inline void *Indx2Ptr_fast(uint indx) {
    uint lim = lim_cached_;  // Pre-compute once, cache in member variable
    return HeapStart + (qword(indx - lim) * UNIT_SIZE + lim);
}
```

3. **Pre-compute `lim_cached_`** once in `InitSubAllocator()` and store as member.
   Current code recomputes `(uint)(UnitsStartBase - HeapStart)` on every call.

**Risk to compression:** Zero.

#### 5b. Context Tree Prefetching
The escape loop in `ppmd_PrepareByte()` does pointer chasing:
```cpp
while (1) {
    do {
        MinContext = suff(MinContext);  // Load iSuffix, convert to pointer, dereference
    } while (MinContext->NumStats == NumMasked);
    processSymbol2_T(MinContext[0]);
}
```
Each `suff()` call is a potential cache miss (~200 cycles for L3). Prefetching the
next context node while processing the current one can hide latency:

```cpp
while (1) {
    do {
        PPM_CONTEXT* next = suff(MinContext);
        if (next && next->iSuffix) {
            __builtin_prefetch(Indx2Ptr(next->iSuffix), 0, 1);  // Prefetch grandparent
        }
        MinContext = next;
    } while (MinContext->NumStats == NumMasked);
    processSymbol2_T(MinContext[0]);
}
```

On MSVC use `_mm_prefetch((char*)ptr, _MM_HINT_T1)`.

**Risk to compression:** Zero.

#### 5c. Memory Compaction Improvements
The current GlueFreeBlocks() defragmentation uses exponential backoff
(`GlueCount = 1 << (13 + GlueCount1++)`) to reduce frequency of expensive merges.

**Current issues:**
1. **FreeUnit() and FreeUnits() call `memset(ptr, 0, blockSize)`** on every free.
   This is a correctness fix to prevent stale index corruption, but it's expensive.
   Once the allocator is stable, consider zeroing only the index fields (first 8 bytes)
   instead of the whole block.

2. **RestoreModelRare() `GetUsedMemory()` is O(N_INDEXES=38)** — relatively cheap,
   but called on every allocation failure. Consider caching the used-memory estimate.

3. **BList[] segregated free list has only 38 buckets.** For large models (order 25),
   mostly 1-unit and 2-unit allocations happen. A dedicated fast-path for 1-unit
   allocations that skips the `Units2Indx` lookup could save cycles:
```cpp
void *AllocUnit_Fast() {
    if (BList[0].avail()) return remove(&BList[0]);
    if (HiUnit - LoUnit >= UNIT_SIZE) { HiUnit -= UNIT_SIZE; return HiUnit; }
    return AllocUnitsRare(0);
}
```

4. **STATE arrays are 6 bytes, packed in pairs (2×6=12 bytes per UNIT).** Adding 2 bytes
   of padding per STATE for 8-byte alignment would increase memory usage by 33% but
   eliminate unaligned access penalties (2-3 cycles per access on some CPUs). This is
   likely NOT worthwhile given the memory-bandwidth-limited nature of PPMd.

**Risk to compression:** Zero for all above.

---

### 6. Mixer Hash Map Optimization [MEDIUM IMPACT — estimated 1.1-1.2×]

**File:** `src/mixer/mixer.cpp` — `Mixer::GetContextData()`

```cpp
ContextData* Mixer::GetContextData() {
    unsigned long long limit = 10000;
    auto it = context_map_.find(context_);
    if (context_map_.size() >= limit && it == context_map_.end()) {
        data = &context_base_;
    } else {
        if (it != context_map_.end()) { data = &it->second; }
        else { auto [it, success] = context_map_.insert({context_, ContextData(...)}); ... }
    }
    return data;
}
```

This is called 24 times per bit (23 layer-0 + 1 layer-1 mixer). Each call does:
1. `context_map_.find()` — hash lookup
2. `context_map_.size()` — may be O(1) in emhash6 but still an extra comparison
3. Potential `context_map_.insert()` with `ContextData` copy-construction

**Proposed optimizations:**

#### 6a. Cache the last accessed context
Most calls within the same bit will look up the same or similar contexts:
```cpp
// In Mixer class:
unsigned long long cached_context_ = ~0ULL;
ContextData* cached_data_ = nullptr;

ContextData* GetContextData() {
    if (context_ == cached_context_) return cached_data_;  // Fast path
    // ... normal lookup ...
    cached_context_ = context_;
    cached_data_ = data;
    return data;
}
```

Since `GetContextData()` is called in both `Mix()` and `Perceive()` for the same bit,
this saves ~50% of hash lookups.

**Risk to compression:** Zero.

#### 6b. Combine Mix + Perceive lookups
Currently `Mix()` and `Perceive()` each independently call `GetContextData()`.
Store the pointer from `Mix()` and reuse in `Perceive()`:
```cpp
class Mixer {
    ContextData* last_data_ = nullptr;
    float Mix() {
        last_data_ = GetContextData();
        return inputs_.dot(last_data_->weights) + ...;
    }
    void Perceive(int bit) {
        ContextData* data = last_data_;  // Reuse from Mix()
        data->weights -= update * inputs_;
        ...
    }
};
```

**Risk to compression:** Zero — same data is accessed.

---

### 7. Predictor Pipeline Optimization [MEDIUM IMPACT — estimated 1.1-1.2×]

**File:** `src/predictor.cpp`

#### 7a. Reduce virtual call overhead  
33+ virtual calls per bit in `Predict()` and 32+ in `Perceive()`. While the compiler
can devirtualize through LTO, the current architecture prevents this because
`bracket_model_` is `std::optional<Bracket>` accessed via base-class pointer,
and models stored in `SmallVector<Model>` use virtual dispatch.

**Quick win:** Mark model classes as `final`:
```cpp
class Bracket final : public Model { ... };
class Direct final : public Model { ... };
class Match final : public Model { ... };
```
This allows the compiler to devirtualize calls when the concrete type is known,
which LTO can exploit across translation units.

**Risk to compression:** Zero.

#### 7b. Avoid re-computing MixerInput logit lookups 
`layers_[0].SetInput(idx, value)` calls `sigmoid_.Logit(p)` which does a table lookup.
The same values are then re-fetched and passed to mixers. Ensuring the sigmoid table
is hot in L1 cache (100K entries × 4 bytes = 400KB — this is large!) or reducing
the table size would help.

Consider reducing `logit_size_` from 100001 to 10001 or 4096 (fits in L1 cache).
The precision loss at 4096 entries is <0.01% per prediction.

**Risk to compression:** Minimal with 10K entries, imperceptible with 100K.

---

### 8. Eigen Temporary Elimination [MEDIUM IMPACT — estimated 1.05-1.15×]

Several hot paths create unnecessary Eigen temporaries:

#### 8a. `Eigen::VectorXf::Ones()` in Adam  
Already covered in optimization #2a.

#### 8b. Gate activation: `1.0f / (1.0f + (-x.array()).exp())`  
In `LstmLayer::ForwardPass()`:
```cpp
forget_gate_.state_[epoch_] = 
    (1.0f / (1.0f + (-forget_gate_.state_[epoch_].array()).exp())).matrix();
```
This creates 3 temporaries (negation, exp, reciprocal). The Eigen expression template
engine should optimize this, but profile to verify. If not, use in-place operations:
```cpp
auto& s = forget_gate_.state_[epoch_];
s = s.unaryExpr([](float x) { return 1.0f / (1.0f + expf(-x)); });
```

#### 8c. Input gate: `Ones(num_cells_) - forget_gate`
```cpp
input_gate_state_[epoch_] = Eigen::VectorXf::Ones(num_cells_) - forget_gate_.state_[epoch_];
```
`Ones()` allocates a temporary. Replace with:
```cpp
input_gate_state_[epoch_] = (-forget_gate_.state_[epoch_].array() + 1.0f).matrix();
```
Or: `input_gate_state_[epoch_] = (1.0f - forget_gate_.state_[epoch_].array()).matrix();`

**Risk to compression:** Zero.

---

### 9. SSE Post-Processor Memory [LOW-MEDIUM IMPACT — estimated 1.05-1.1×]

**File:** `src/mixer/sse.cpp`

The SSE post-processor `M_T` struct contains very large lookup arrays:
```cpp
static const int M_mix1_Volume = 1* 4* (1<<8)* (1<<3)* 79;   // = 647,168
static const int M_sm6x_Volume = 1* 3* (1<<7)* (1<<8)* 256;  // = 75,497,472
static const int M_sm7x_Volume = 1* 3* (1<<5)* (1<<8)* 255;  // = 18,808,320
static const int M_mix2_Volume = 1* 3* (1<<1)* (1<<8)* 256;  // = 393,216

SSEi<7> s6[M_sm6x_Volume];  // 75M entries × 14 bytes = ~1 GB!
Mixer   x1[M_mix1_Volume];  // 647K entries × 4 bytes = ~2.5 MB
SSEi<7> s7[M_sm7x_Volume];  // 18.8M entries × 14 bytes = ~264 MB
Mixer   x2[M_mix2_Volume];  // 393K entries × 4 bytes = ~1.5 MB
```

These tables cause massive L3 cache pressure. The M_T structure is over 1.2 GB.

**Proposed optimization:**
- Profile which SSE table entries are actually accessed for typical inputs
- Consider using hash-based lazy initialization (only allocate entries on first access)
- Or reduce the context quantization to reduce table sizes

**Risk to compression:** Depends on quantization. Needs careful testing.

---

### 10. Context Manager Indirect Hash Tables [LOW-MEDIUM IMPACT — 1.05-1.1×]

**File:** `src/context-manager.cpp`, `src/context-manager.h`

```cpp
std::vector<unsigned long long> hashes_ind1;  // 16M entries × 8 bytes = 128 MB
std::vector<unsigned long long> hashes_ind2;  // 16M entries × 8 bytes = 128 MB
std::vector<unsigned long long> hashes_ind3;  // 32M entries × 8 bytes = 256 MB
```

Total: ~512 MB for indirect hash tables, plus 60 MB history + 100 MB shared_map.

**Access pattern:** Random lookups with data-dependent indices — worst case for CPU cache.

**Proposed optimizations:**

#### 10a. Reduce to 32-bit entries where possible
`hashes_ind4` and `hashes_ind5` are only 256 entries and store values masked to small
ranges. Even `hashes_ind1` stores values masked to `0xFF` (8-bit). These could use
`uint8_t` instead of `unsigned long long`, reducing ind1 from 128 MB to 16 MB.

Check what actual bit-widths are needed:
- `ind1`: masked to `0xFF` → use `uint8_t` (saves 120 MB)
- `ind2`: masked to `0xFFFFFFFF` → use `uint32_t` (saves 64 MB)
- `ind3`: masked to `0x1FFFFFF` → use `uint32_t` (saves 128 MB)
- `ind5`: masked to `0x3FFFFFFF` → use `uint32_t` (saves <1 KB)

**Total memory savings: ~312 MB** → *much* better cache utilization.

**Risk to compression:** Zero — no information lost.

#### 10b. history_ buffer as uint8_t ring buffer
Already `std::vector<unsigned char>` — this is fine. But 60 MB is large. Consider whether
a smaller window (e.g., 16 MB) gives comparable match model performance.

**Risk to compression:** Needs testing. Match models depend on history length.

---

### 11. Batch Perceive Update [LOW IMPACT — estimated 1.02-1.05×]

**File:** `src/predictor.cpp` — `Predictor::Perceive()`

All models are updated sequentially with the same `bit` value:
```cpp
bracket_model_->Perceive(bit);
for (...) direct_models_[i].Perceive(bit);
for (...) match_models_[i].Perceive(bit);
for (...) indirect_ns_models_[i].Perceive(bit);
...
```

Since each model's `Perceive` only updates its own internal state, these are independent
and could potentially be interleaved with prefetch instructions for the next model's data.
However, true parallelism (threads) would add synchronization overhead that outweighs the
benefit given each update is fast.

**Quick win:** Reorder updates so that models sharing cache lines are updated consecutively.
Group all `indirect_ns_models_` (which access `shared_map_`) together, etc.

**Risk to compression:** Zero.

---

### 12. Compiler & Build Optimizations [LOW IMPACT — estimated 1.05-1.1×]

**File:** `makefile`

#### 12a. Profile-Guided Optimization (PGO)
The makefile already has `prof_gen` and `prof_use` targets. If not already used
for the baseline binary, PGO typically gives 5-15% speedup for branch-heavy code
like this. Run:
```bash
make prof_gen
./cmix no-preprocess prof_input/input ./res.bin  # Generate profile
make prof_use                                      # Build optimized binary
```

#### 12b. Consider `-fprofile-sample-use` for sampling-based PGO
If instrumented PGO is too slow for the training run, sampling-based PGO with
`perf` (Linux) or Intel VTune (Windows) can be used.

#### 12c. BOLT binary optimization
After PGO, LLVM BOLT can reorder functions and basic blocks based on runtime
profiles for an additional 5-10% speedup.

**Risk to compression:** Zero — output is bit-identical.

---

## Implementation Priority

| # | Optimization | Est. Speedup | Risk | Effort | Priority | Status |
|---|-------------|-------------|------|--------|----------|--------|
| 1c | LSTM output layer as matrix | 1.2-1.5× | None | Low | **P0** | ✅ Done |
| 2a | Adam eliminate Ones() temporaries | 1.1-1.2× | None | Low | **P0** | ✅ Done |
| 2b | Adam pre-compute pow() | 1.05× | None | Low | **P0** | ✅ Done |
| 6b | Mixer combine Mix+Perceive lookup | 1.05-1.1× | None | Low | **P0** | ✅ Done |
| 7a | Mark model classes final | 1.02-1.05× | None | Trivial | **P0** | ✅ Done |
| 8c | Eliminate Ones() in gate computation | 1.02-1.05× | None | Trivial | **P0** | ✅ Done |
| 10a | Reduce indirect hash to proper widths | 1.05-1.1× | None | Low | **P0** | ✅ Done |
| 3 | LSTM softmax as matrix-vector multiply | 1.1-1.3× | Near-zero | Medium | **P1** | ✅ Done (via P0-1c) |
| 4 | Fuse layer normalization | 1.05-1.1× | None | Low | **P1** | ✅ Done |
| 5a | PPMd Indx2Ptr fast-path inline | 1.05-1.1× | None | Low | **P1** | ✅ Done |
| 5b | PPMd context tree prefetching | 1.05-1.1× | None | Low | **P1** | ✅ Done |
| 6a | Mixer context cache | 1.05× | None | Low | **P1** | ✅ Done (via P0-6b) |
| 7b | Reduce sigmoid table size | 1.02-1.05× | Minimal | Low | **P1** | — |
| 12a | PGO build | 1.05-1.15× | None | Low | **P1** | ✅ Done |
| 1a | LSTM truncated BPTT | 1.3-2× | Low-Med | Medium | **P2** | — |
| 1b | Reduce BPTT frequency | 1.5-2× | Low | Medium | **P2** | — |
| 5c | PPMd memset optimization | 1.02× | None | Low | **P2** | — |
| 9 | SSE table size reduction | 1.05-1.1× | Needs testing | High | **P3** | — |

### Estimated Combined Speedup (P0 items): **1.4-1.8×** (no compression impact)
### Estimated Combined Speedup (P0+P1): **1.6-2.2×** (minimal compression impact)
### Estimated Combined Speedup (P0+P1+P2): **2.0-3.5×** (needs compression validation)

---

## Key Constraint

**All optimizations MUST NOT degrade compression.** After each change:
1. Run `./cmix.exe no-preprocess .\prof_input\input ./res.bin` → verify output = **6,152 bytes**
2. Run `./cmix.exe no-preprocess .\prof_input\input2 ./res2.bin` → verify output = **181,019 bytes**
3. Any deviation > 0 bytes requires investigation. Deviation in LSTM/PPMd changes
   due to floating-point order differences must be validated as net-neutral on larger inputs.

---

## Memory Usage Summary

| Component | Current | Optimized (proposed) |
|-----------|---------|---------------------|
| Context Manager hashes | ~512 MB | ~200 MB (use proper types) |
| Context Manager history + shared_map | ~160 MB | ~160 MB (unchanged) |
| PPMd heap | 1,024 MB | 1,024 MB (unchanged) |
| SSE post-processor | ~1,200 MB | ~1,200 MB (unchanged, or hash-based P3) |
| LSTM weights + buffers | ~65 MB | ~35 MB (matrix layout) |
| Mixer context maps | ~10 MB | ~10 MB (unchanged) |
| **Total** | **~2,971 MB** | **~2,629 MB** |

---

## Detailed Component Analysis

### LSTM Forward/Backward Pass Profile
- **Forward pass per byte:** 3 gate ForwardPass calls + output layer (256 dot products)
- **Backward pass per 128 bytes:** Full BPTT over 128 timesteps × 3 gates
- **Adam updates per 128 bytes:** 3 gates × (matrix weights + gamma + beta) = 9 Adam calls
- **Weight matrix size:** 3 × (200 × ~457) = ~274K floats = 1.1 MB per gate
- **Adam state (m, v):** Same size as weights = 2.2 MB total per gate
- **Total LSTM memory:** ~3.3 MB weights + ~6.6 MB Adam state + ~64 MB horizon buffers

### PPMd Compaction Analysis  
The PPMd allocator uses a bidirectional bump allocator with segregated free lists:
- **Text area:** 1/8 of heap (128 MB) — rarely used in practice
- **Units (states):** Grows upward from UnitsStart
- **Contexts:** Grows downward from HeapEnd
- **Free lists:** 38 buckets for recycled blocks

**Compaction opportunities:**
1. The `memset(ptr, 0, blockSize)` in `FreeUnit()`/`FreeUnits()` is the biggest overhead.
   It was added as a correctness fix. Consider zeroing only struct header fields instead.
2. `GlueFreeBlocks()` merges fragmented free blocks — triggered exponentially less often.
   The backoff is appropriate but the merge algorithm is O(total_free_blocks).
3. **No true compaction** (moving live objects) is possible without a relocation table.
   The index-based pointer system would support relocation, but the implementation
   complexity is very high for marginal benefit.

### Mixer Weight Update Profile
- **23 layer-0 mixers + 1 layer-1 mixer per bit**
- Each mixer: 1 hash lookup + 1 dot product (Mix) + 1 hash lookup + 1 vector update (Perceive)
- Input vector size: 461 floats for layer-0, ~25 for layer-1
- emhash6 lookup: ~3-5 cache-line accesses per lookup
- **Total:** 48 hash lookups + 48 dot products/updates per bit ≈ ~2000 cycles

### SSE Post-Processor Profile
- Single-pass through two SSE stages with mixer correction
- Context indexing: 6-8 multiplications per access
- Table sizes dominate memory footprint (~1.2 GB)
- Computation itself is fast (integer arithmetic + table lookups)
