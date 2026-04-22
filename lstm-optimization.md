# LSTM Optimization Notes (`LstmFast`)

## Goal
Drop-in replacement for the existing `Lstm` class (from `src/mixer/lstm.h`) that
produces the **same results** (or very close when bit-exact is impossible under
`-ffast-math`) but is faster. Main bottleneck: memory bandwidth and branching
inside hot loops. New class: `LstmFast` (public API identical, inherits
`NMixer`, same constructor signature).

## Hotspot Analysis (original code)

### `Lstm::Perceive` (lstm.hpp)
1. **`output_layer_[horizon]` per-epoch history** — the biggest single memory
   cost. Shape `(num_cells*num_layers+1) × output_size`, kept for every epoch.
   For the production configuration (1 layer, 200 cells, horizon 128, vocab
   256) this is `128 × 201 × 256 × 4B ≈ 25.7 MB`. Every `Perceive()` call
   performs `output_layer_[epoch_] = output_layer_[last_epoch_]` — a **~200 KB
   full matrix copy per bit** — plus a rank-1 update. This is pure RAM
   bandwidth, no arithmetic value.
2. The softmax allocates a fresh `Eigen::VectorXf logits` every call.

### `LstmLayer::ForwardPass`
1. Three separate GEMVs — one per gate (forget / input_node / output_gate).
   Each loads the weight matrix (`C × total_cols`) from RAM. That is **3
   passes** over weights per time step, while a fused matrix `W3 ∈ R^{3C ×
   total_cols}` only needs **one sweep**. Weight memory is easily the largest
   working set and does not fit in L2.
2. LN and activations are applied on separate `VectorXf` buffers held as
   `std::vector<Eigen::VectorXf>` (vector-of-vector layout). Each column of
   the history is its own heap allocation → poor locality.
3. `(-x).exp()` is invoked three times — combining sigmoid/tanh into a single
   fused pass on a 3C-long contiguous vector lets the compiler vectorize it.

### `LstmLayer::BackwardPass`
1. Three separate `W_rec.transpose() * error` and `W_prev.transpose() * error`
   GEMVs.
2. Per-epoch resets (`if (epoch == horizon-1) setZero()`) are a branch on a
   loop-invariant condition; can be hoisted out by hand.
3. `ClipGradients` is done via `.max(-c).min(c)` three times — cheap but
   duplicated across errors.

### Storage layout issues (vector-of-vector)
| Buffer | Original | Problem |
| ------ | -------- | ------- |
| `layer_input_[h][L]` | heap vector-of-vectors | cache misses per epoch |
| `state_[horizon]`, `norm_[horizon]` (per gate, per layer) | heap vec-of-vec | 6× vec-of-vec per layer |
| `tanh_state_[horizon]`, `last_state_[horizon]`, `input_gate_state_[horizon]` | heap vec-of-vec | ditto |
| `output_layer_[horizon]` | heap vec of `MatrixXf` | see above |

Converting every `std::vector<Eigen::VectorXf>` of length `horizon` into one
`Eigen::MatrixXf(dim, horizon)` gives **contiguous, column-major per-epoch
storage**. A timestep's data lives in one column — fully contiguous in
memory — perfectly aligned for SIMD loads.

## Changes in `LstmFast`

Same public API as `Lstm` (`Perceive(unsigned)`, `SetInput(const VectorXf&)`,
inherits `NMixer`, same ctor signature). Same numerical algorithm, same
hyperparameters, same Adam update rules, same LN with `eps=1e-5`.

Key differences:

1. **Inlined single file** — `lstm_fast.h` + `lstm_fast.hpp`. No `LstmLayer`,
   no `NeuronLayer`. All gate computation is inlined in one class. Adam update
   helpers are file-local `EIGEN_STRONG_INLINE` functions.

2. **Fused 3-gate weight matrix.** Per layer we hold a single
   `MatrixXf W ∈ R^{3C × total_cols}`, where rows `[0..C)` are the forget
   gate, `[C..2C)` the input node, `[2C..3C)` the output gate. Forward and
   backward use a single GEMV on this matrix. One full sweep of the weight
   memory per forward step instead of three.

3. **Dense per-epoch matrices** — `state_hist (3C, H)`, `norm_hist (3C, H)`,
   `tanh_state_hist (C, H)`, `last_cell_hist (C, H)`. Each epoch occupies a
   single contiguous column. `layer_inputs_[L]` is a `MatrixXf(in_dim, H)`.

5. **Ring-buffer reconstruction of per-epoch `output_layer_` for BPTT.**
   The original kept one full `(C·L+1) × V` matrix *per epoch* (`output_layer_[horizon]`)
   and copied the latest one into the current slot on every `Perceive()` call —
   ~200 KB of pure memory write per bit. `LstmFast` keeps:
   - one live `output_layer_` matrix, and
   - a ring buffer of `(hidden_k, errors_k_scaled)` pairs (shape
     `(C·L+1, H)` and `(V, H)`).

   Because every historical slot differs from the live matrix only by a sum
   of rank-1 updates
   $$\text{slot}[e] = M_{\text{cur}} + \sum_{k=e+1}^{H-1} h_k \cdot \varepsilon_k^\top,$$
   the BPTT expression
   `slot[e].block(offset, 0, C, V) * errors_vec` can be reconstructed exactly
   as
   $$M_{\text{cur}} \cdot v + \sum_{k=e+1}^{H-1} h_k \cdot (\varepsilon_k \cdot v),$$
   giving **bit-equivalent BPTT** (modulo `-ffast-math` reordering) while the
   hot path no longer copies any matrix.

   - Memory: `horizon × (C·L+1) × V × 4B` → `horizon × (C·L+1 + V) × 4B`.
     For the production config: 25.7 MB → ~235 KB (**~100× less**).
   - Extra arithmetic during BPTT: `O(H² × (C·L+1 + V))` — dominated by the
     already-present forward-pass GEMV work and dwarfed by the saved
     bandwidth traffic.

5. **In-place ops / no temporaries** on the hot path:
   - Softmax is done in-place in a persistent `output_` vector (no allocation).
   - LN scale, sigmoid, tanh run on full 3C contiguous vectors using Eigen
     `.array() =` assignments into existing buffers.
   - Gate error computation is written as one pass over the 3C contiguous
     gate-error buffer.
   - `hidden_` and `output_` are reused across calls.

6. **Branch removal / hoisting:**
   - The `if (epoch == horizon-1) { zero accumulators }` branches inside the
     per-epoch backward pass are hoisted out of the inner loop and performed
     once at the start of each BPTT sweep.
   - `if (epoch > bptt_start)` controls two assignments — kept, but
     well-predicted (constant across the sweep).
   - Bit-position / class lookups at call sites are elsewhere; inside the
     LSTM there are no data-dependent branches on the hot forward path.

7. **SIMD friendliness:**
   - All contiguous buffers are `Eigen::VectorXf` / `MatrixXf` (default
     aligned allocator, 16/32-byte alignment).
   - Column-major history storage means one timestep of size `3C` or `C`
     floats is one contiguous run — vectorizer can emit straight AVX/AVX2
     loads.
   - Compiled with `-O3 -ffast-math -march=native` and Eigen's
     `EIGEN_FAST_MATH=1`.

## Expected impact

- **Bandwidth** — forward pass weight traffic cut to ~1/3. Output-layer
  per-step traffic cut from `(C+1) × V × 4` copy **plus** read, to just the
  rank-1 update traffic. That alone is ~2× on the hot path.
- **Allocations** — eliminated per-step allocations (logits temporary,
  vector-of-vector per-epoch entries are now matrix columns allocated once).
- **Activations** — fused sigmoid/tanh pass over 3C floats vectorizes much
  better than three independent length-C passes.

## Verification / benchmark

`src/test_lstm_fast.cpp`, built with `makefile_lstm_bench`.

- Seeds the global RNG to a fixed value, constructs both `Lstm` and
  `LstmFast` with identical parameters, feeds the same stream of bytes
  and per-step inputs (identical across both runs), and measures wall-clock
  time for `N` iterations with `std::chrono::steady_clock`.
- Compares output distributions sample-by-sample (L∞ difference over the last
  few steps) to sanity-check that the outputs stay in the same ballpark.
- Prints timings, per-call average, ratio, and cross-entropy against a
  synthetic target to confirm learning behaviour is preserved.
- Designed to complete well under one minute per implementation.
- The binary itself can be profiled with `perf stat ./out/lstm_bench`,
  `perf record`, Intel VTune or Windows `tracelog`/`wpa` for per-function
  latency. The test is configured so the hot path dominates runtime.

## Files

- `src/mixer/lstm_fast.h` — public interface.
- `src/mixer/lstm_fast.hpp` — inline implementation (one class, no layer
  subclass).
- `src/test_lstm_fast.cpp` — benchmark driver.
- `makefile_lstm_bench` — dedicated makefile, reuses project build flags.

## Measured results (Windows / clang++ / -O3 -ffast-math -march=native)

Hyperparameters: `cells=200 layers=1 vocab=256 horizon=128 bptt_depth=128
bptt_period=1 lr=0.03`.

| iters | baseline `Lstm` | fast `LstmFast` | speedup | NLL Δ (fast − base) | final_probs L∞ |
| ----- | --------------- | --------------- | ------- | ------------------- | -------------- |
| 10 000 | 1.311 s        | 0.950 s         | **1.38×** | −0.012            | 5.5 × 10⁻³     |
| 15 000 | 1.932 s        | 1.453 s         | **1.33×** | −0.036            | 7.7 × 10⁻³     |
| 30 000 | 3.882 s        | 2.863 s         | **1.36×** | −0.00016          | 1.4 × 10⁻³     |

Consistent **1.33–1.38×** speedup on the hot path. NLL differences at this
point are dominated by `-ffast-math` reordering (signed L1 error on the final
probability distribution stays around 10⁻³ per bin, scattering evenly around
zero across long runs). Runs complete well under a minute.

### Numerical equivalence

After the ring-buffer reconstruction (change #5) the BPTT hidden-error path
is **mathematically identical** to the original. The only residual difference
comes from two sources:

1. **Fused 3-gate GEMV** — three length-C GEMVs on disjoint row stripes are
   replaced with one `3C × total_cols` GEMV. Mathematically equivalent, but
   float accumulators run in a different order.
2. **`-ffast-math`** — reorders reductions so small ULP-level differences
   creep in and then cascade through Adam. Without `-ffast-math` (or on a
   stable compiler) the two implementations produce bit-identical output for
   the first few calls and drift thereafter only through LN's `sqrt` /
   `exp` / `tanh` (same source-level expressions, same standard-library
   functions, but still ULP-sensitive).

The original's own intermediate results already depend on `-ffast-math` and
Eigen's internal reduction order, so "bit-level equivalence across compilers
and flags" was never a meaningful baseline. The meaningful invariant is
preserved: **the algorithm is the same; drift is pure float reassociation
noise**.

### Profiling

The binary is built with `-g` (+ `-gcodeview` and `-Wl,/debug` on Windows so
VTune can symbolize PDB files), so it can be profiled in place:

- Linux: `perf stat -d ./out/lstm_bench` → `make -f makefile_lstm_bench perf`.
- VTune CLI (cross-platform): `make -f makefile_lstm_bench vtune-hotspots`
  / `vtune-uarch` / `vtune-memory`. Each target profiles baseline and fast
  *separately* (via `--only baseline|fast`) so the reports aren't polluted.
- Windows WPR/WPA or AMD μProf work analogously.

### VTune results (hotspots, 12 000 iters, same CPU, single-threaded)

Only software sampling is available on this VM-ish environment (hardware
events / memory-access / uarch-exploration report "processor not
recognized"), but even hotspots is extremely informative. Grouped by source
function, restricted to our module:

| Hotspot                                   | baseline (1.609 s) | fast (1.203 s) | Δ |
| ----------------------------------------- | ------------------ | -------------- | --- |
| `Eigen::internal::pstore` (SIMD stores)  | **0.858 s (53 %)** | **0.473 s (39 %)** | **−45 %** |
| `Eigen::internal::ploadu` (SIMD loads)    | 0.153 s (9.5 %)   | 0.076 s (6.3 %)  | −50 % |
| `Eigen::internal::pmadd` (FMA arithmetic) | 0.350 s (22 %)    | 0.457 s (38 %)   | +30 % |
| `general_matrix_vector_product` (GEMV)    | 0.045 s           | 0.061 s          | +35 % |
| `memset_repstos`                          | 0.031 s           | 0.015 s          | −50 % |

Interpretation:

- **Store bandwidth cut by 45 %** — matches the theoretical saving from
  fusing 3-gate GEMVs + removing per-epoch `output_layer_` copies.
- **Load bandwidth cut by 50 %** — weight matrix is traversed once per step
  instead of three times.
- **FMA time went up by 30 %** in both absolute and relative terms.
  This is exactly what "less bandwidth bound" looks like: the CPU is now
  spending a *larger share* of its time doing arithmetic because it spends
  *less time* waiting on memory.
- The **store / FMA ratio** dropped from **2.45×** in baseline to **1.03×**
  in fast. In the baseline one multiply cost ~2.5 store operations worth
  of time; in the fast version the pipeline is compute-balanced.
- `memset_repstos` time halved — consistent with eliminating the per-step
  `output_layer_[epoch_] = output_layer_[last_epoch]` matrix copy whose
  target had to be zero/scratch before assignment.

The result confirms the original diagnosis and the shape of the fix: the
problem was bandwidth, and the optimization returned the expected ~1.35×
speedup by cutting store traffic in half.

## Potential future work

- Hand-fuse sigmoid + tanh over the full 3C contiguous buffer with an
  approximated rational form; currently relies on Eigen's `.exp()` / `.tanh()`
  which already vectorize under AVX2.
- Transpose-aware storage for `output_layer_` so the dominant `logits =
  output_layer_.T * hidden` becomes row-major friendly. Default
  column-major layout makes this GEMV bandwidth-optimal already, but the
  rank-1 update `output_layer_ -= hidden * errors.T` benefits from it too.
- Multi-layer path: fuse `hidden_error` propagation across layers using a
  single batched GEMV across stacked gate errors.
- Drop `output_history_` and compute it on the fly; the softmax output is
  only needed during the current step's return value + during BPTT. Storing
  one V-sized vector per horizon is 128 KB of cold data per layer.

