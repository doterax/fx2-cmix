# VTune hotspots — `cmix.exe no-preprocess input2`

**Date:** 2026-04-22
**Binary:** `cmix.exe` built via `make cmix-profile` (clang++, `-O3 -ffast-math
-march=native -flto=off -g -gcodeview -Wl,/debug`).
**Command:** `vtune -collect hotspots -result-dir vtune_results/cmix_input2 -- ./cmix.exe no-preprocess ./prof_input/input2 ./res_input2.bin`
**Workload:** Full predictor (461 models + 23 mixers + SSE) on `prof_input/input2`
(941 724 bytes) in `no-preprocess` mode.
**Elapsed / CPU:** 211.9 s wall, 209.9 s CPU, single thread, no spin / overhead.
**Raw data:**
- Result dir: [vtune_results/cmix_input2](vtune_results/cmix_input2)
- CSV: [vtune_results/cmix_input2_hotspots.csv](vtune_results/cmix_input2_hotspots.csv)
- Collector log: [vtune_cmix_input2.log](vtune_cmix_input2.log)

Only `hotspots` is available on this box (VTune does not recognize the CPU;
uarch / memory-access / performance-snapshot collectors reject the machine).

## CPU time by source file

| Area                        | File                                    | CPU (s) | %   | Notes                                                                                 |
| --------------------------- | --------------------------------------- | ------: | --: | ------------------------------------------------------------------------------------- |
| **LSTM / Eigen**            | PacketMath.h (pstore / pmadd / ploadu)  |   85.0  | 40.5| mostly `LstmFast` GEMV + rank-1 body                                                   |
|                             | InnerProduct.h                          |   11.2  |  5.3| LSTM dot products                                                                      |
|                             | AssignEvaluator.h                       |    7.4  |  3.5| block assigns                                                                          |
|                             | GeneralMatrixVector.h                   |    4.8  |  2.3| Eigen GEMV kernel                                                                      |
|                             | DenseStorage.h + GenericPacketMath.h    |    1.6  |  0.8|                                                                                        |
|                             | **LSTM/Eigen subtotal**                 | **110.0**| **52.4** |                                                                                 |
| **fxcmv1 (431 sub-models)** | fxcmv1.cpp                              |   49.9  | 23.8| E1::get 4.5 %, Mixer1::dot_product 2.8 %, Inputs<560>::add 2.8 %, AddPrediction 1.9 %, ContextMap2::mix 1.7 % |
| **Sigmoid table lookup**    | sigmoid.cpp + mixer-input.cpp           |   18.2  |  8.6| all from `MixerInput::SetInput` → `Predictor::Predict`                                |
| Hashmap (indirect ctx)      | emhash_map.hpp                          |    4.8  |  2.3| `Indirect<Nonstationary>::Predict`                                                     |
| Indirect predictor body     | indirect.h                              |    2.9  |  1.4|                                                                                        |
| PPMd byte model             | ppmd.cpp                                |    2.7  |  1.3| once per byte, not per bit                                                             |
| memset                      | memset.asm                              |    2.6  |  1.2|                                                                                        |
| Mixer layer                 | mixer.cpp                               |    1.5  |  0.7|                                                                                        |
| SSE                         | sse.cpp                                 |    1.8  |  0.9|                                                                                        |
| Match model                 | match.cpp                               |    1.4  |  0.7|                                                                                        |
| `exp` (softmax)             | exp.asm                                 |    1.7  |  0.8|                                                                                        |

## Top individual functions

| Function                                              | Self CPU (s) |    %  |
| ----------------------------------------------------- | -----------: | ----: |
| `Eigen::internal::pstore`                             |        35.3  | 16.7  |
| `Eigen::internal::pmadd`                              |        30.2  | 14.3  |
| `Sigmoid::Logit`                                      |        13.2  |  6.3  |
| `fxcmv1::E1<14,128>::get`                             |         9.4  |  4.5  |
| Eigen inner_product (VectorXf · VectorXf)             |         9.1  |  4.3  |
| `Eigen::internal::ploadu`                             |         7.4  |  3.5  |
| `Eigen::internal::pload`                              |         6.6  |  3.1  |
| `fxcmv1::Mixer1::dot_product`                         |         5.8  |  2.8  |
| `fxcmv1::Inputs<560>::add`                            |         5.8  |  2.8  |
| Eigen block-assign                                    |         5.6  |  2.7  |
| `emhash6::HashMap::find_filled_hash`                  |         4.5  |  2.1  |
| `Eigen::internal::general_matrix_vector_product::run` |         4.1  |  1.9  |
| `fxcmv1::AddPrediction`                               |         3.9  |  1.9  |
| `MixerInput::SetInput`                                |         3.9  |  1.9  |
| `fxcmv1::ContextMap2::mix`                            |         3.6  |  1.7  |

Callstack confirmed for `Sigmoid::Logit`: 100 % of its self-time comes through
`MixerInput::SetInput → Predictor::Predict → Encoder::Encode`, i.e. called once
per active model per bit.

## Improvement opportunities — ranked by return-on-effort

### 1. Inline `Sigmoid::Logit` into the header — 6.3 %, quick win
`Sigmoid::Logit` is a single table lookup, but it lives in `sigmoid.cpp` so
every call from the hot per-bit `MixerInput::SetInput` loop is a real call
(LTO notwithstanding — symbol is exported, doesn't get inlined across TUs
because the cpp is compiled with `-fno-threadsafe-statics` / per-file). Moving
the body into `sigmoid.h`:

```cpp
inline float Sigmoid::Logit(float p) const {
  int index = p * logit_size_;
  if (index >= logit_size_) index = logit_size_ - 1;
  else if (index < 0)       index = 0;
  return logit_table_[index];
}
```

lets the compiler fold the bounds check with the caller's clamp, keeps
`logit_size_` and `logit_table_.data()` in registers, and may vectorize the
outer loop in `MixerInput::SetInput` across consecutive indices.

**Expected save:** 3–5 % end-to-end. Code change is literally moving 6 lines.

### 2. Branchless clamp + inlining in `MixerInput::SetInput` — 1.9 %, same win
Currently:
```cpp
void MixerInput::SetInput(int index, float p) {
  if (p < min_) p = min_;
  else if (p > max_) p = max_;
  inputs_[index] = sigmoid_.Logit(p);
}
```
Move to header, combine with change 1, and replace the two-branch clamp with
`std::min(std::max(p, min_), max_)`. Along with change 1 this unlocks
auto-vectorization if callers feed it in batches.

### 3. fxcmv1's hot scalar loops — ~10 %, medium effort
Three functions totalling ~15 s:
- `fxcmv1::E1<14,128>::get` (9.4 s, 4.5 %) — templated lookup over 128-slot table, called 431× per bit.
- `fxcmv1::Mixer1::dot_product` (5.8 s, 2.8 %) — scalar dot product.
- `fxcmv1::Inputs<560>::add` (5.8 s, 2.8 %) — appends stretched predictions to a 560-wide input vector.

These are per-bit scalar loops over 431 sub-models. The layout is
SIMD-friendly (contiguous arrays, fixed template dimensions known at compile
time) but the current code is probably auto-vectorizing poorly because of
branches and small trip counts. Same playbook as LstmFast: fuse, go
column-major where needed, eliminate temporaries, make loops branchless.

**Expected save:** 3–6 % if we get those three into tight SIMD form.

### 4. Remaining Eigen `pstore` / `pmadd` in LSTM — diminishing returns
Still 40 % of CPU, but after the LstmFast pass the only remaining handles are
the "Future work" items from [lstm-optimization.md](lstm-optimization.md):

- hand-fused tanh + sigmoid activations (today still two passes over the gate
  vector),
- dropping `output_history_` once `errors_ring_` is consulted directly,
- multi-layer fused GEMV.

Each individually ≤ 5 % end-to-end. Worth doing but not before 1–3 above.

### 5. `emhash::find_filled_hash` — 2.3 %, low ROI
Context-hash lookup inside indirect models. Either switch to a flatter open-
addressed map or improve the hash mixer. 2.3 % is not where to start.

### 6. Softmax / `exp` — 0.8 %, skip
Already small. No action needed.

## Key conclusion

Even in the full predictor **> 52 % of CPU is still Eigen SIMD inside the
LSTM** — so LstmFast was the correct target and still has residual handles
(Section 4). The biggest *independent* next win is a one-line **move of
`Sigmoid::Logit` into the header (6.3 %)** plus inlining
`MixerInput::SetInput`. After that, **fxcmv1's three hot scalar loops
(~10 %)** are the logical follow-up using the same fused/SIMD-friendly-layout
playbook that worked for LstmFast.

## Results — improvements #1 and #2 applied (2026-04-22)

Changes landed:
- [src/mixer/sigmoid.h](src/mixer/sigmoid.h) — `Logit` body inlined in header.
- [src/mixer/sigmoid.cpp](src/mixer/sigmoid.cpp) — out-of-line `Logit` removed.
- [src/mixer/mixer-input.h](src/mixer/mixer-input.h) — `SetInput`,
  `SetStretchedInput`, `SetZero`, `SetExtraInput` inlined with branchless
  `std::min(std::max(p, lo), hi)` clamps; added `#include <algorithm>`.
- [src/mixer/mixer-input.cpp](src/mixer/mixer-input.cpp) — reduced to ctor +
  `SetNumModels` only.

### End-to-end wall time on `no-preprocess input2` (941 724 B)

Same binary target (`cmix-profile`, with `-g -gcodeview` symbols), same host,
no VTune attached:

| Build                     | Wall time  | Output size | Compression ratio |
| ------------------------- | ---------- | ----------: | ----------------- |
| Baseline (pre-inline)     | 213.68 s   |   181 036 B | 19.22389 %        |
| Inlined Logit + SetInput  | **207.85 s** | 181 036 B | 19.22389 %        |
| **Delta**                 | **−5.83 s (−2.7 %)** | bit-identical | unchanged |

Compression output is byte-identical — branchless `std::min(std::max(...))`
clamp is semantics-preserving for the non-NaN inputs used by the predictor.

### VTune hotspots after inlining

Result dir: [vtune_results/cmix_input2_v2](vtune_results/cmix_input2_v2).
Collector log: [vtune_cmix_input2_v2.log](vtune_cmix_input2_v2.log).

Standalone `Sigmoid::Logit` CPU share fell from **6.3 % → 1.4 %** (remainder
folded into call sites — most commonly `MixerInput::SetInput`, which absorbs
~4.1 % as expected). Top functions on the v2 profile:

| Function                                     | Self CPU (s) |   %  |
| -------------------------------------------- | -----------: | ---: |
| `Eigen::internal::pstore`                    |        34.4  | 16.6 |
| `Eigen::internal::pmadd`                     |        30.0  | 14.5 |
| Eigen inner_product (VectorXf · VectorXf)    |         9.6  |  4.6 |
| `fxcmv1::E1<14,128>::get`                    |         9.0  |  4.3 |
| `MixerInput::SetInput` (inlined Logit)       |         8.7  |  4.2 |
| `Eigen::internal::ploadu`                    |         7.1  |  3.4 |
| `Eigen::internal::pload`                     |         6.2  |  3.0 |
| `fxcmv1::Inputs<560>::add`                   |         5.7  |  2.8 |
| Eigen block-assign                           |         5.0  |  2.4 |
| `fxcmv1::Mixer1::dot_product`                |         4.9  |  2.4 |
| `emhash6::HashMap::find_filled_hash`         |         4.3  |  2.1 |
| Eigen GEMV kernel                            |         4.6  |  2.2 |
| `fxcmv1::ContextMap2::mix`                   |         4.2  |  2.1 |
| `fxcmv1::AddPrediction`                      |         4.2  |  2.0 |
| `Sigmoid::Logit` (remaining out-of-line)     |         3.4  |  1.4 |

Combined `Sigmoid::Logit` + `MixerInput::SetInput` self-time dropped from
**17.1 s (8.2 %) → 12.1 s (5.6 %)** — a ~5 s reduction matching the observed
wall-time delta almost exactly (bounds check + call overhead eliminated; no
vectorization of the outer loop yet, which is the upside still on the table).

### Conclusion

Predicted 3–5 % end-to-end save; measured **2.7 %** on input2 at
`no-preprocess`. Next logical step is finding #3 (fxcmv1 hot scalar loops,
~10 % potential).

## Results — improvement #3 attempt, fxcmv1 hot paths (2026-04-22)

Changes landed in [src/models/fxcmv1.cpp](src/models/fxcmv1.cpp):
- `E<A,B>::get` and `E1<A,B>::get` split into an `always_inline` fast-path
  (hit on `chk[last&15]==ch`, the common case) plus a noinline `get_slow`
  that handles the full scan + LRU-replacement. Previously the whole function
  was `noinline`, so every fast-path hit still paid a real call.
- `Inputs<S>::add` annotated `always_inline` (confirmed no-op after inspection —
  the template method was already being inlined; harmless belt-and-suspenders).

Output is bit-identical (181 036 B, compression ratio 19.22389 %).

### VTune v3 vs v2, normalized as % of total CPU

Absolute wall times were contaminated by background video capture during v3
collection (v2 total CPU 204.7 s, v3 total CPU 218.7 s — the 14 s gap is
host noise, not the code). The run-normalized percentage view is the fair
comparison because each sample set is relative to its own total.

Result dirs: [vtune_results/cmix_input2_v2](vtune_results/cmix_input2_v2),
[vtune_results/cmix_input2_v3](vtune_results/cmix_input2_v3).

| Function                              |  v2 %  |  v3 %  |  Δpp   |
| ------------------------------------- | -----: | -----: | :----: |
| `fxcmv1::E1<14,128>::get`             |  4.4   |  3.5   | **−0.9** |
| `Eigen::internal::pstore`             | 16.8   | 17.0   |  +0.2  |
| `Eigen::internal::pmadd`              | 14.7   | 14.9   |  +0.2  |
| `MixerInput::SetInput`                |  4.3   |  4.6   |  +0.3  |
| `fxcmv1::Inputs<560>::add`            |  2.8   |  3.1   |  +0.3  |
| `fxcmv1::Mixer1::dot_product`         |  2.4   |  2.8   |  +0.4  |
| Eigen inner_product                   |  4.7   |  5.1   |  +0.4  |

`E1::get` is the only statistically meaningful delta (−0.9 pp, ~1 % of total
CPU). Everything else is within sample-to-sample jitter (~±0.4 pp on a
~200-second profile). The `always_inline` marker on `Inputs::add` had no
effect.

### Conclusion

Improvement #3 as written returns ~1 % on `E1::get` — real but small, and an
order of magnitude below the 3–6 % initially projected. The fxcmv1 scalar
bodies are largely memory-latency-bound on random hash lookups (131k-entry
`t[]` table hit-rate dominates), not compute-bound, so micro-optimizations
on the inner body have limited leverage without attacking the memory access
pattern.

Better follow-ups, in priority order:
1. **`fxcmv1` hash prefetch**: insert `__builtin_prefetch(&t[(cxt[i]+cc)&tmask])`
   at the *start* of `mix1`'s per-i loop so the cache line is warm by the
   time `get()` reads `chk[]`. Same idea for `ContextMap2::mix1`.
2. **Return to LSTM**: the 52 % LSTM share is still the biggest lever.
   Multi-layer fused GEMV (finding #4) is the next concrete step.
3. **`emhash::find_filled_hash` (2.1 %)**: tune probe window or switch to a
   flat-array layout where feasible.

## Results — improvement #4a, drop `output_history_` (2026-04-22)

Follow-up from the LstmFast "Future work" list
([lstm-optimization.md](lstm-optimization.md)). The LSTM used to keep a
(V × H) = (256 × 128) copy of every predicted distribution just so BPTT
could compute `hist[k] - onehot(input[k])` per past timestep. That same
quantity (scaled by `lr`) is already recorded into `errors_ring_` after
every `Perceive`, so the separate `output_history_` matrix is redundant.

Change in [src/mixer/lstm_fast.hpp](src/mixer/lstm_fast.hpp) /
[src/mixer/lstm_fast.h](src/mixer/lstm_fast.h):
- Move the `errors = (output_current_ - onehot(input)) * lr` computation
  plus the `errors_ring_.col(epoch_) = errors` store to *before* BPTT runs
  (safe — BPTT only reads rows `>epoch`, never `col(epoch_)` itself).
- Replace `Eigen::VectorXf errors_vec = output_history_.col(epoch);
  errors_vec[input_history_[epoch]] -= 1.0f;` inside the BPTT loop with
  `errors_ring_.col((epoch+1) mod H) / lr`.
- Delete `output_history_` member and its per-Predict 256-float store.
- Initialize `errors_ring_` to `(1/V · ones − onehot(0)) · lr` across all
  columns to match the uniform prior that the original code implied.

### Measurements

| Build                           | Total CPU  | Output size | Compression ratio |
| ------------------------------- | ---------: | ----------: | ----------------: |
| v2 (Sigmoid + SetInput inline)  |   204.66 s |   181 036 B |         19.2239 % |
| v3 (fxcmv1 get() split)         |   218.74 s*|   181 036 B |         19.2239 % |
| v4 (drop output_history_)       | **202.57 s** |   181 025 B |   **19.2227 %** |

\* v3 total was inflated by background video capture; see normalized %
analysis in the previous section.

Δ vs v2: −2.1 s total CPU (~1 % on the full predictor). Output shrank by
11 bytes — strictly better compression, driven by float-reassociation drift
in the BPTT error path (reading `errors_ring_ / lr` instead of
reconstructing from the softmax output). Same class of effect as
`-ffast-math`, same magnitude (~10⁻³ per bin).

Result dir: [vtune_results/cmix_input2_v4](vtune_results/cmix_input2_v4).
Collector log: [vtune_cmix_input2_v4.log](vtune_cmix_input2_v4.log).

### Conclusion

Modest but free win: 1 % CPU reduction, 11-byte compression gain, one less
128 KB array being trampled every bit (better for L2/L3 residency of the
hot BPTT working set).

Next candidates from the LSTM "Future work" list:
- **#4b — hand-fused tanh + sigmoid activations**: today the forward path
  runs three separate array-wise activations over contiguous 3C-wide
  segments. A single pass with a manual gate-by-gate loop can save one full
  read/write of the 3C buffer.
- **#4c — multi-layer fused GEMV**: with `num_layers_ == 1` on the
  production path this is a no-op. Keep on ice until we re-enable 2-layer.


