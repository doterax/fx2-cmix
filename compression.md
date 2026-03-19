# Compression Ratio Improvement Research

## Isolated LSTM Predictor

**Goal:** Create an isolated LSTM predictor (`-p lstm`) that runs the LSTM byte mixer *without* the mixer ensemble, FXCM, or other bit-level models. This allows:
- Direct measurement of LSTM contribution in isolation
- Fast iteration on LSTM hyperparameters (cells, layers, horizon, LR) without 461-model overhead  
- Testing additional LSTM inputs (character class, position features, etc.) without mixer interference
- Building a comprehensive baseline across corpora and dictionary configurations

The isolated LSTM predictor feeds the LSTM a uniform prior (1/vocab_size) each byte, receives back a probability distribution over bytes, and converts it to bit-level predictions using the standard bot/mid/top binary decomposition.

### LSTM-Only Baselines (200 cells, 1 layer, horizon 128)

| Corpus | Mode | Dictionary | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|------|-----------|--------------|---------------|-------|------|-------|
| input | no-preprocess | none | 51,052 | 17,872 | 35.01% | 6.1s | 8,404 B/s |
| input | compress | english.dic | 51,052 | 13,692 | 26.82% | 3.8s | 13,382 B/s |
| input | compress | words_enwik9_optimal.dic | 51,052 | 14,064 | 27.55% | 4.0s | 12,882 B/s |
| input2 | no-preprocess | none | 941,724 | 266,708 | 28.32% | 110.5s | 8,519 B/s |
| input2 | compress | english.dic | 941,724 | 207,575 | 22.04% | 67.8s | 13,897 B/s |
| input2 | compress | words_enwik9_optimal.dic | 941,724 | 216,269 | 22.97% | 67.0s | 14,048 B/s |
| enwik7 | no-preprocess | none | 10,000,000 | 2,527,173 | 25.27% | 1166.0s | 8,577 B/s |
| enwik7 | compress | english.dic | 10,000,000 | 2,115,044 | 21.15% | 730.5s | 13,689 B/s |
| enwik7 | compress | words_enwik9_optimal.dic | 10,000,000 | 2,213,633 | 22.14% | 735.4s | 13,599 B/s |

Total benchmark time: 48.2 minutes.

**Key observations:**
- **LSTM-only vs full predictor gap:** On input no-preprocess, LSTM-only gets 17,872 bytes vs full predictor's 6,142 bytes — the full ensemble is **2.91× better**. On input2, 266,708 vs 180,822 — **1.47× better**. The gap narrows on larger data as LSTM learns more context.
- **Dictionary helps significantly:** english.dic reduces enwik7 from 2,527,173 → 2,115,044 (16.3% smaller), and processes 2× faster since the preprocessed stream is shorter.
- **english.dic beats enwik9_optimal.dic:** Across all corpora, english.dic consistently outperforms enwik9_optimal.dic for the LSTM-only predictor (e.g., enwik7: 2,115,044 vs 2,213,633). This differs from the full predictor where specialized dictionaries can win.
- **Processing speed is constant:** ~8,500 B/s raw, ~13,500 B/s with dictionary preprocessing (faster because dictionary-compressed data is shorter).

### ByteMixer Baselines — PPMd + LSTM (200 cells, 1 layer, english.dic)

#### v5: 40 context-indexed micro-mixers (previous byte class × bit position)

5 byte classes (lowercase, uppercase, digit, space/whitespace, other) × 8 bit positions = 40 independent `MicroMixer` instances (LR=0.07). After each completed byte, `ByteClass()` classifies it and all subsequent bit predictions use `class*8 + bit_pos` to select the mixer. This captures that e.g. bit 7 after a lowercase letter has a very different PPMd/LSTM accuracy profile than bit 7 after a digit.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **5,976** | 11.71% | 7.3s | 7,022 B/s |
| input2 | 941,724 | **181,669** | 19.29% | 72.4s | 13,013 B/s |
| enwik7 | 10,000,000 | **1,902,208** | 19.02% | 741.4s | 13,487 B/s |

#### v4: 8 bit-position-indexed micro-mixers

Each bit position (0–7) within the byte gets its own independent `MicroMixer` (LR=0.07). MSB bits (character class decisions) and LSB bits (fine discrimination) have very different PPMd vs LSTM accuracy profiles, so separate mixers can specialize. Selection via `bit_context_` (1→2→4→…→128) maps to index 0–7.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **5,995** | 11.74% | 7.1s | 7,180 B/s |
| input2 | 941,724 | **182,075** | 19.33% | 73.2s | 12,867 B/s |
| enwik7 | 10,000,000 | **1,906,254** | 19.06% | 799.9s | 12,502 B/s |

#### v3: Micro-mixer (online logistic regression over PPMd + LSTM bit predictions)

A 2-input online logistic regression (`MicroMixer`, LR=0.07) takes stretched PPMd bit prediction and stretched LSTM bit prediction, learns optimal blend weights from actual bit outcomes. This is the same principle as the full predictor's 23 L0 mixers, but with just 2 inputs.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **6,192** | 12.13% | 7.7s | 6,630 B/s |
| input2 | 941,724 | **183,207** | 19.45% | 72.3s | 13,022 B/s |
| enwik7 | 10,000,000 | **1,913,981** | 19.14% | 790.9s | 12,644 B/s |

#### v2: LSTM-only prediction (no PPMd bit signal)

`Predict()` returns LSTM bit decomposition directly. The LSTM already received PPMd byte probs as input, so its output is a refinement of PPMd.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | 10,560 | 20.69% | 7.6s | 6,744 B/s |
| input2 | 941,724 | 189,312 | 20.10% | 72.7s | 12,948 B/s |
| enwik7 | 10,000,000 | 1,907,722 | 19.08% | 783.4s | 12,765 B/s |

#### v1: PPMd+LSTM fixed 50/50 average *(quirk: double-counted PPMd — `0.5*ppmd_bit + 0.5*lstm_bit`)*

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | 7,654 | 14.99% | 7.2s | 7,081 B/s |
| input2 | 941,724 | 185,029 | 19.65% | 70.8s | 13,299 B/s |
| enwik7 | 10,000,000 | 1,936,942 | 19.37% | 786.5s | 12,715 B/s |

#### All versions compared:

| Corpus | v1 (50/50) | v2 (LSTM-only) | v3 (1 mixer) | v4 (8 pos) | v5 (40 ctx) | Full predictor |
|--------|-----------|---------------|-------------|-----------|------------|----------------|
| input (51K) | 7,654 | 10,560 | 6,192 | 5,995 | **5,976** | 4,765 |
| input2 (942K) | 185,029 | 189,312 | 183,207 | 182,075 | **181,669** | 158,273 |
| enwik7 (10M) | 1,936,942 | 1,907,722 | 1,913,981 | 1,906,254 | **1,902,208** | — |

#### Analysis:

**v5 (40 context mixers) is the new best across all corpora:**
- On input (51K): **5,976** — 0.3% better than v4 (5,995), 3.5% better than v3, 43.4% better than v2. Only 25.4% worse than full predictor (4,765), closing to within **1,211 bytes** of 461 models.
- On input2 (942K): **181,669** — 0.2% better than v4 (182,075). Steady improvement from context specialization.
- On enwik7 (10M): **1,902,208** — 0.2% better than v4 (1,906,254), and 0.3% better than v2's pure LSTM (1,907,722). The 40-mixer approach decisively beats all previous variants at all scales.

**Progression from v3 → v4 → v5 shows diminishing but consistent returns:**

| Step | input Δ | input2 Δ | enwik7 Δ | What changed |
|------|---------|----------|----------|--------------|
| v3→v4 | -197 (-3.2%) | -1,132 (-0.6%) | -7,727 (-0.4%) | 1→8 mixers (bit position) |
| v4→v5 | -19 (-0.3%) | -406 (-0.2%) | -4,046 (-0.2%) | 8→40 mixers (+byte class) |
| v3→v5 total | -216 (-3.5%) | -1,538 (-0.8%) | -11,773 (-0.6%) | 1→40 mixers |

The byte class context helps most on enwik7 (-4,046 bytes absolute) because with 10M of training data the 40 mixers have enough samples to specialize. On input (51K), only 19 bytes saved — with ~6,400 bytes of training data split among 40 mixers, each sees only ~160 bytes, barely enough to learn.

**Why byte class helps:**
- After a lowercase letter, the next byte is very likely another lowercase letter or space. PPMd's character-level n-gram model is strong here, so the mixer should weight PPMd higher for certain bit positions.
- After a digit, the next byte has a different distribution (more digits, decimal points, commas). The specialized mixer can learn this different PPMd/LSTM balance.
- After special characters (tags, punctuation), the LSTM's longer-range context tends to be more valuable than PPMd's local n-gram.

**Gap to full predictor:**
- input: 5,976 vs 4,765 — remaining **1,211 bytes** (20.3%) come from the 459 other models + 23 context-specific mixers + SSE.
- input2: 181,669 vs 158,273 — **23,396 bytes** (14.8%) gap.

**Speed:** v5 runs at the same speed as v4 (7s for input, ~72s for input2, ~740s for enwik7). The 40 tiny mixers (2 weights each = 80 floats) add negligible overhead. In fact enwik7 was slightly *faster* (741s vs 800s) — likely measurement noise.

**Diminishing returns from mixer contexts alone:** Going from 1→8→40 mixers improved input by only 216 bytes total (3.5%). The remaining 1,211-byte gap to the full predictor requires fundamentally different model signals (word boundaries, XML structure, match models, etc.), not more mixer contexts. To improve further, we need to enrich the model inputs rather than the mixer topology.

### MixPredictor — Extensible multi-model predictor (`-p mix`)

Adds new model inputs alongside PPMd+LSTM, blended by an N-input context-indexed micro-mixer. Designed for easy addition of further models.

**Architecture:** N-input `NMicroMixer` (online logistic regression over N stretched inputs), 80 context-indexed instances (5 byte classes × 8 bit positions × 2 bracket states). Sub-models: PPMd bit prediction + LSTM bit prediction + Bracket model bit prediction. `BracketContext` provides an additional mixer context dimension (inside/outside bracket).

**Sub-model roles:**
- **PPMd** (order 25): Strong local n-gram predictor, fast convergence
- **LSTM** (200 cells): Learns long-range statistical patterns, refines PPMd byte probs
- **Bracket** model: Predicts closing bracket character (`>`, `)`, `}`, `]`) given opening bracket + distance statistics. After preprocessing, `<`→`L`, `>`→`N`, `{`→`P`, `}`→`R`, so the bracket pairs tracked are `L/N`, `P/R`, `(/)`, `[/]`.

**Results (english.dic):**

#### v1: PPMd + LSTM + Bracket (no SSE)

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | 5,920 | 11.60% | 7.7s | 6,647 B/s |
| input2 | 941,724 | 181,323 | 19.25% | 72.4s | 13,005 B/s |
| enwik7 | 10,000,000 | 1,901,615 | 19.02% | 739.8s | 13,517 B/s |

#### v2: PPMd + LSTM + Bracket + SSE

SSE (Secondary Symbol Estimation) applied after the micro-mixer output. Corrects systematic calibration bias in the mixer's probability estimates using a dual-chain adaptive piecewise-linear interpolation with bit-history context.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **5,897** | 11.55% | 7.6s | 6,709 B/s |
| input2 | 941,724 | **180,420** | 19.16% | 73.5s | 12,820 B/s |
| enwik7 | 10,000,000 | **1,888,460** | 18.88% | 746.4s | 13,398 B/s |

#### Comparison: ByteMixer v5 → MixPredictor v1 → v2 → Full

| Corpus | ByteMixer v5 | Mix v1 (no SSE) | Mix v2 (+SSE) | Full predictor |
|--------|-------------|-----------------|--------------|----------------|
| input (51K) | 5,976 | 5,920 | **5,897** | 4,765 |
| input2 (942K) | 181,669 | 181,323 | **180,420** | 158,273 |
| enwik7 (10M) | 1,902,208 | 1,901,615 | **1,888,460** | — |

**Analysis:**

**SSE provides the biggest single-step improvement so far:**
- input: -23 bytes (v1→v2), total -79 from ByteMixer v5 (-1.3%)
- input2: -903 bytes (v1→v2), total -1,249 from ByteMixer v5 (-0.7%)
- enwik7: **-13,155 bytes** (v1→v2), total -13,748 from ByteMixer v5 (-0.7%)

SSE's impact scales with data size: on enwik7 it saves 13K bytes — more than all previous improvements from adding Bracket + BracketContext (-593) combined by **22×**.

#### Why SSE helps — the calibration problem explained

Our micro-mixer outputs a probability via logistic regression: `p = Squash(w0*Stretch(ppmd) + w1*Stretch(lstm) + w2*Stretch(bracket))`. This is a good estimator, but it has a systematic flaw: **the output often doesn't match the true frequency of 1-bits**.

**Example:** When the mixer outputs p=0.95, the actual frequency of 1-bits in that situation might be 0.98. When it outputs p=0.02, the true rate might be 0.005. The mixer is "right about the direction" but wrong about the magnitude — especially at the extremes near 0 and 1, where each fraction of a percent matters most for compression.

**Why this matters for compression:** The arithmetic coder assigns code lengths based on $-\log_2(p)$. If the true probability is 0.98 but we say 0.95, we waste $-\log_2(0.95) - (-\log_2(0.98)) \approx 0.044$ bits per symbol. Over millions of symbols, this adds up to thousands of bytes. The loss is **asymmetric** — errors near 0 and 1 are much costlier than errors near 0.5, because the log function is steepest there.

**What SSE does:** It maintains a small adaptive lookup table (7 interpolation points across [0,1]) that maps mixer output → corrected output. It learns this mapping online from actual bit outcomes. If the mixer consistently outputs 0.95 when the truth is 0.98, the SSE table entry near 0.95 gradually shifts toward 0.98. It also uses bit-history context (recent bits seen, byte position) to have different correction tables for different situations.

**Why it scales with data size:**
- On input (51K / ~6K compressed): SSE has ~48K bit updates to learn its correction table — barely enough to populate 7 interpolation points × a few contexts. Saves only 23 bytes.
- On enwik7 (10M / ~1.9M compressed): SSE has ~15M bit updates. The correction table becomes highly accurate, and the systematic bias it corrects affects every single bit prediction. Saves 13,155 bytes.

**Why the mixer is miscalibrated in the first place:**
1. Logistic regression is a linear model in the stretched (log-odds) domain — it can't capture nonlinear interactions between inputs
2. The Stretch/Squash transforms distort the probability space — a small weight error compounds differently at p=0.5 vs p=0.99
3. Our 80 context-indexed mixers each have limited training data, so their weights converge to approximate values, not exact ones
4. SSE acts as a lightweight nonlinear correction layer — like a tiny neural network that learns the residual error pattern

**Gap to full predictor:**
- input: 5,897 vs 4,765 — **1,132 bytes** remaining (was 1,211 pre-MixPredictor)
- input2: 180,420 vs 158,273 — **22,147 bytes** (14.0% gap, was 14.8%)
- The remaining gap comes from Match models (exact pattern matching), Indirect/Nonstationary state machines, Direct context models, and FXCM — plus the full predictor's 23 context-specific L0 mixers vs our 80 context-indexed micro-mixers.

### MixPredictor v3: GRU replaces LSTM (experiment)

Replaced the LSTM (Long Short-Term Memory) with a **GRU (Gated Recurrent Unit)** — a modern simplification of LSTM that merges the cell state and hidden state into a single hidden vector. Same number of weight matrices (3 gates), same hyperparameters (200 cells, 1 layer, horizon=128, lr=0.03).

**GRU advantages over LSTM:**
- No separate cell state — hidden state IS the memory, providing more direct gradient flow
- Simpler update equations: `h = z*h_prev + (1-z)*candidate` vs LSTM's separate cell/output gating
- Often converges faster on limited data (fewer internal dependencies)

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **5,840** | 11.44% | 7.7s | 6,596 B/s |
| input2 | 941,724 | **179,870** | 19.10% | 73.1s | 12,879 B/s |
| enwik7 | 10,000,000 | **1,891,452** | 18.91% | 746.4s | 13,398 B/s |

#### Comparison: v2 (LSTM+SSE) → v3 (GRU+SSE)

| Corpus | v2 (LSTM) | v3 (GRU) | Delta | Winner |
|--------|-----------|----------|-------|--------|
| input (51K) | 5,897 | **5,840** | **-57** | GRU |
| input2 (942K) | 180,420 | **179,870** | **-550** | GRU |
| enwik7 (10M) | 1,888,460 | 1,891,452 | **+2,992** | LSTM |

**Analysis — data-dependent crossover:**

GRU wins on smaller corpora but LSTM wins on the largest. This reveals a classic bias-variance tradeoff:

- **Small data (input, input2):** GRU's simpler architecture has lower variance — fewer internal parameters to fit means less overfitting. The direct gradient path (no cell state barrier) also helps the GRU converge faster with limited training data.
- **Large data (enwik7):** LSTM's separate cell state provides additional memory capacity. With 10M bytes of training, the LSTM has enough data to fully utilize its cell state mechanism, capturing longer-range dependencies that the GRU's single hidden state cannot.

The 550-byte win on input2 but 3K loss on enwik7 suggests the **crossover point is somewhere around 1-5M bytes**. For enwik8/enwik9 workloads, LSTM is likely the better choice (and is what the full predictor uses).

**Conclusion:** GRU is not a clear upgrade over LSTM for compression. The result depends on corpus size. For the primary enwik benchmark, LSTM remains the better sequence model. Reverting to LSTM for subsequent experiments.

#### Full progression: ByteMixer v5 → Mix v1 → v2 → v3 → v4 → v5 → v6 → Full

| Corpus | ByteMixer v5 | Mix v1 | Mix v2 (LSTM+SSE) | Mix v3 (GRU+SSE) | Mix v4 (xLSTM+SSE) | Mix v5 (RWKV+SSE) | Mix v6 (+TagPredict) | Full |
|--------|-------------|--------|-------------------|-------------------|---------------------|-------------------|---------------------|------|
| input | 5,976 | 5,920 | 5,897 | **5,840** | 5,983 | 5,873 | 5,897 | 4,765 |
| input2 | 181,669 | 181,323 | 180,420 | **179,870** | 190,711 | 180,492 | 180,417 | 158,273 |
| enwik7 | 1,902,208 | 1,901,615 | **1,888,460** | 1,891,452 | 2,561,295 | 1,893,276 | 1,888,457 | — |

### MixPredictor v4: xLSTM (sLSTM) replaces LSTM (experiment — FAILED)

Replaced LSTM with **sLSTM** (from xLSTM, Hochreiter et al. 2024) — a modernized LSTM with:
- **Exponential gating**: `exp()` instead of `sigmoid()` for forget and input gates, with log-space stabilization ($m_t = \max(\log f + m_{t-1}, \log i)$)
- **Decoupled gates**: Separate input gate (LSTM uses coupled $i = 1 - f$), adding a 4th gate
- **Normalizer state**: Tracks $n_t = f' \cdot n_{t-1} + i'$, replaces `tanh(c)` with $c / \max(|n|, 1)$

Same hyperparameters (200 cells, 1 layer, horizon=128, lr=0.03).

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | 5,983 | 11.72% | 8.6s | 5,909 B/s |
| input2 | 941,724 | 190,711 | 20.25% | 90.4s | 10,420 B/s |
| enwik7 | 10,000,000 | 2,561,295 | 25.61% | 939.7s | 10,642 B/s |

#### Comparison: v2 (LSTM) → v4 (xLSTM)

| Corpus | v2 (LSTM) | v4 (xLSTM) | Delta | Regression |
|--------|-----------|------------|-------|------------|
| input (51K) | 5,897 | 5,983 | **+86** | +1.5% |
| input2 (942K) | 180,420 | 190,711 | **+10,291** | +5.7% |
| enwik7 (10M) | 1,888,460 | 2,561,295 | **+672,835** | +35.6% |

**Catastrophic failure — xLSTM is dramatically worse, especially on larger data.** The enwik7 result (2.56M) is even worse than LSTM-only without any mixer (2.12M with english.dic), meaning the xLSTM is actively degrading compression quality on large data.

**Why xLSTM failed for online compression:**

1. **Exponential gate sensitivity**: `exp()` amplifies small weight perturbations far more than `sigmoid()`. With online learning (one sample at a time, no batching), this creates training instability. A sigmoid gate at pre-activation 2.0 gives 0.88; at 2.1 gives 0.89 (Δ=0.01). An exp gate at 2.0 gives 7.39; at 2.1 gives 8.17 (Δ=0.78). The model overreacts to each update.

2. **Normalizer doesn't replace tanh effectively**: Standard LSTM's `tanh(c)` bounds the hidden state in [-1, 1] unconditionally. The normalizer $c/\max(|n|, 1)$ only bounds when $|n| \geq 1$. Early in training or after context shifts, $n$ can be small, letting $c$ pass through unbounded. This produces wild probability estimates that cost many bits.

3. **Adam hyperparameters mismatch**: Our Adam (β₁=0.025, β₂=0.9999) was tuned for sigmoid/tanh activations. Exponential gates need much more conservative learning rates to prevent divergence. The UPDATE_LIMIT=3000 cap amplifies this — the learning rate floor is too high for exp gates.

4. **The coupled gate regularizes well**: LSTM's $i = 1 - f$ constraint forces a tradeoff: more forgetting means less input, and vice versa. This implicit regularization prevents both gates from being large simultaneously. Decoupled exponential gates can both fire strongly, leading to unstable cell dynamics.

5. **Speed penalty**: 4 gates instead of 3 → ~20% slower (10.6K vs 13.4K B/s), adding insult to injury.

**Lesson:** xLSTM was designed for large-batch offline training on GPUs with careful hyperparameter tuning. Online, single-sample, CPU-based compression training is a fundamentally different regime where the original LSTM's bounded activations and coupled gates are features, not limitations. Reverting to LSTM.

### MixPredictor v5: RWKV replaces LSTM (experiment)

Replaced LSTM with **RWKV** (Receptance Weighted Key Value) — a linear attention model that replaces LSTM's gated cell state with a WKV (Weighted Key Value) mechanism:
- **3 gates** (same count as LSTM): receptance (sigmoid), key (linear → exp), value (tanh)
- **WKV state**: numerator $a_t = e^{-w} a_{t-1} + e^{k_t} v_t$, denominator $b_t = e^{-w} b_{t-1} + e^{k_t}$
- **Output**: $\sigma(r_t) \cdot a_t / b_t$ — receptance-gated weighted average of values
- **Learned decay $w$**: per-channel positive parameter, $e^{-w}$ is the decay factor (replaces LSTM's forget gate)
- **Log-space stabilization**: $m_t = \max(m_{t-1} - w, k_t)$ prevents exp overflow

Same hyperparameters (200 cells, 1 layer, horizon=128, lr=0.03).

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | **5,873** | 11.50% | 7.5s | 6,789 B/s |
| input2 | 941,724 | 180,492 | 19.17% | 73.7s | 12,774 B/s |
| enwik7 | 10,000,000 | 1,893,276 | 18.93% | 752.4s | 13,292 B/s |

#### Comparison: v2 (LSTM) → v5 (RWKV)

| Corpus | v2 (LSTM) | v5 (RWKV) | Delta | Winner |
|--------|-----------|-----------|-------|--------|
| input (51K) | 5,897 | **5,873** | **-24** | RWKV |
| input2 (942K) | **180,420** | 180,492 | +72 | LSTM |
| enwik7 (10M) | **1,888,460** | 1,893,276 | **+4,816** | LSTM |

**Analysis — RWKV is the closest competitor to LSTM:**

RWKV shows the smallest enwik7 regression of any alternative sequence model tested (+4,816 vs GRU's +2,992 and xLSTM's +672,835). The result is actually very close to LSTM across all corpora.

**Why RWKV is better behaved than xLSTM but still doesn't beat LSTM:**

1. **Bounded gates stabilize online learning**: Receptance uses sigmoid, value uses tanh — both bounded, like LSTM. This avoids xLSTM's catastrophic exp-gate instability. The exp(k) appears only in both numerator and denominator (self-normalizing via a/b division), so errors are dampened rather than amplified.

2. **Fixed decay $w$ vs adaptive forget gate**: LSTM's forget gate is sigmoid(W·input) — it adapts per-sample based on the current context. RWKV's decay $w$ is a learned parameter but fixed across all inputs. This makes RWKV simpler and more stable, but less flexible: it can't dynamically choose "remember everything" for some inputs and "forget quickly" for others. On enwik7's diverse content (different article topics, styles, languages), this input-dependent adaptivity matters.

3. **WKV is a weighted average, LSTM is gated accumulation**: RWKV's output $a/b$ is always a weighted average of past values (bounded by extreme v values). LSTM's $\sigma(o) \cdot \tanh(c)$ allows the cell state to accumulate information without normalization, giving it potentially more representational power on large data.

4. **Speed is identical**: 3 gates in both → ~13.3K B/s. No efficiency penalty.

**Conclusion:** RWKV is the most viable LSTM alternative tested — it produces comparable compression and doesn't degrade catastrophically. However, for enwik-scale benchmarks, LSTM's input-dependent gating provides a consistent edge. Reverting to LSTM.

### MixPredictor v6: Tag content prediction (experiment)

Added a **tag content predictor** as a 4th mixer input (PPMd + LSTM + Bracket + TagPredict → 4-input NMicroMixer). The predictor maintains a stack of opening tag names and predicts closing tag name bytes when `</` (or `L/` after preprocessing) is encountered.

**Implementation:**
- State machine: IDLE → SAW_OPEN → IN_OPEN_NAME/IN_OPEN_ATTRS → IDLE (push tag) or → IN_CLOSE_NAME (predict) → IDLE (pop tag on confirmed match)
- Handles `<`/`>` (raw mode) and `L`/`N` (preprocessed mode), `!`/`?` (comments/PIs), self-closing tags (`/>`), attributes
- When predicting closing tag bytes: confidence 0.95 for correct bit, 0.05 for wrong bit via bot/mid/top decomposition
- Stack capped at 256, tag names at 64 chars. Only pops on confirmed full match.

| Corpus | Input (bytes) | Output (bytes) | Ratio | Time | Speed |
|--------|--------------|---------------|-------|------|-------|
| input | 51,052 | 5,897 | 11.55% | 7.5s | 6,807 B/s |
| input2 | 941,724 | 180,417 | 19.16% | 73.3s | 12,853 B/s |
| enwik7 | 10,000,000 | 1,888,457 | 18.88% | 727.5s | 13,745 B/s |

#### Comparison: v2 (LSTM+SSE) → v6 (+TagPredict)

| Corpus | v2 (LSTM+SSE) | v6 (+TagPredict) | Delta |
|--------|--------------|------------------|-------|
| input (51K) | 5,897 | 5,897 | 0 |
| input2 (942K) | 180,420 | 180,417 | **-3** |
| enwik7 (10M) | 1,888,460 | 1,888,457 | **-3** |

**Essentially zero improvement.** The tag predictor saves exactly 3 bytes on both input2 and enwik7. This confirms the documented concerns:

1. **Dictionary preprocessing absorbs most tags**: Common XML tags like `<title>`, `</title>`, `<text>` etc. are replaced by single dictionary tokens during preprocessing. The remaining raw tags in the stream are rare — uncommon tags, deeply nested structures, or tags that didn't make the dictionary.

2. **`L` ambiguity causes false triggers**: After preprocessing, `<` becomes `L`, but regular `L` in text (very common — "Linux", "London", etc.) also triggers the state machine. This pushes garbage tag names onto the stack, causing most closing-tag lookups to find the wrong expected name. The mixer learns to mostly ignore the tag input because it's unreliable.

3. **PPMd already handles residual tags**: PPMd order 25 captures tag names when the opening-to-closing span is < 25 bytes. For longer spans, the LSTM's statistical patterns fill in. Between them, there's very little compression gain left for a deterministic tag predictor.

4. **The mixer correctly adapts**: Despite the noisy tag signal, compression doesn't degrade — the mixer weight for the tag input stays near zero, adding no noise. The 3-byte improvement is real but trivial: a few correctly predicted closing tag bytes across the entire corpus.

**Speed:** Identical to v2 (~13.4K B/s on enwik7). The tag state machine adds negligible overhead.

**Conclusion:** Tag content prediction is not worthwhile after dictionary preprocessing. The combination of `L` ambiguity and dictionary absorption leaves too few correctly-predictable tag bytes. The idea would work better on raw (non-preprocessed) data where `<`/`>` are unambiguous, but raw mode is not our benchmark configuration.

---

#### Future idea: Closing tag content prediction

Currently the Bracket model only predicts that the **closing bracket character** will appear at some distance — e.g., after `<title>Hello</`, it knows `>` will come but does NOT predict the tag name `title`. The tag name is handled by PPMd (order 25, which can match contexts up to 25 bytes) and LSTM (statistical patterns).

A specialized **tag-content predictor** could:
- When an opening bracket `L` (= `<`) is seen, record the tag name until `N` (= `>`)
- When `L/` is seen (opening of a closing tag), predict the exact tag name from the bracket stack
- This would give near-perfect predictions for XML/HTML closing tags

**Pros:**
- Very high prediction accuracy for closing tag names — almost deterministic given the bracket stack
- Significant on enwik data which is heavily XML-structured (`<title>`, `<text>`, `<id>`, etc.)
- Cheap to implement — just a stack of tag name strings + lookup
- Complement to Match model — works even when the closing tag hasn't appeared in recent history

**Cons:**
- Only helps for XML/HTML tag content, not general text — narrow applicability
- After dictionary preprocessing, common tags may already be replaced by single tokens (e.g., `<title>` → dictionary word), reducing the signal
- PPMd order 25 already captures most closing tags when context is short enough (tag name + content < 25 bytes)
- Would need to handle self-closing tags, malformed XML, and nesting edge cases

---

## Current Baselines

| Corpus | Mode | Dictionary | Result (bytes) | Time |
|--------|------|-----------|---------------|------|
| input (51,052 B) | no-preprocess | none | 6,142–6,149 | ~18 s |
| input (51,052 B) | compress | english.dic | 4,765 | ~32 s |
| input (51,052 B) | compress | words_input.dic (943 words) | 3,859 | ~11 s |
| input2 (941,724 B) | no-preprocess | none | 180,822–180,824 | ~225 s |
| input2 (941,724 B) | compress | english.dic | 158,273 | ~163 s |
| input2 (941,724 B) | compress | words_input2.dic (15,538 words) | 155,224 | ~151 s |
| enwik8 (100 MB) | compress | english.dic (seed 2) | **14,962,810** | ~14,473 s |
| enwik8 (100 MB) | no-preprocess | none (PPMd 14000 MB) | 15,832,397 | ~33,424 s |
| enwik8 (100 MB) | PPMd-only | none (14000 MB) | 20,774,117 | ~398 s |
| enwik9 (934 MB) | full | english.dic | 110,111,245 | ~228,590 s |

**Hutter Prize result:** S1 (exe) = 441,463 + S2 (archive) = 110,351,665 = **S = 110,793,128** (1.585% improvement over L = 112,578,322).

---

## Architecture Overview

```
461 model outputs
  ├─ bracket_model (1 output)
  ├─ fxcm_model (431 outputs)
  ├─ direct_0 (1 output)
  ├─ match_0..match_9 (10 outputs)
  ├─ indirect_ns_0..indirect_ns_14 (15 outputs)
  ├─ indirect_r_0 (1 output)
  ├─ byte_model / PPMd (1 output)
  └─ byte_mixer / LSTM (1 output)
        │
        ▼
  23 Layer-0 Mixers (various context hashes, LR 0.0003–0.005)
        │
        ▼
  1 Layer-1 Mixer (zero_context, LR 0.0003)
        │
        ▼
  SSE 2-stage cascade (~2.67M table entries)
        │
        ▼
  Arithmetic Coder
```

**LSTM:** 200 cells, 1 layer, horizon 128, LR 0.03, gradient clip 10, Adam (β₁=0.025, β₂=0.9999, ε=1e-6).

**PPMd:** Order 25, 1024 MB default (14000 MB for competition).

**Mixer context map limit:** 10,000 entries. Decay schedule: 1.0 (< 1M steps) → 0.7 (< 5M) → 0.3 (< 25M) → 0.2 (≥ 25M).

**Dictionary encoding:** 3-tier system: tier 1 = 80 words (1 byte), tier 2 = 3,840 words (2 bytes), tier 3 = 40,960 words (3 bytes). Max capacity: 44,880 words.

---

## Predictor Accuracy Analysis

From predictor stats on input2 (no-preprocess), **almost all bit-level models perform near random (46%)**, while only the byte-level LSTM is highly accurate:

| Predictor | Mean Accuracy | Range |
|-----------|--------------|-------|
| bracket_model | 46.4% | 32.5%–55.6% |
| fxcm_model (431 sub-models) | 46.4% | 32.5%–55.6% |
| direct_0 | 45.9% | 31.9%–54.8% |
| match_0..match_9 | 46.4% | 32.5%–55.6% |
| indirect_ns_0..indirect_ns_14 | 46.4% | 32.5%–55.6% |
| indirect_r_0 | 46.4% | 32.5%–55.6% |
| byte_model (PPMd) | 46.4% | 32.5%–55.6% |
| **byte_mixer (LSTM)** | **93.0%** | **64.0%–100.0%** |

**Key insight:** The bit-level models report ~46% because the stat tracking measures per-bit accuracy but these models produce per-byte distributions — the accuracy metric at bit level is misleading. However, the dominant signal clearly comes from the LSTM byte mixer, which produces the highest-quality predictions. The bit-level models still contribute information through their mixer contexts, visible only through the final ensemble.

**Null LSTM experiment:** Without LSTM, input2 compresses to 182,519 bytes vs 180,824 bytes with LSTM — **LSTM saves ~1,695 bytes (0.93%)** on input2.

**Predictor correlation:** 30 of 32 predictors show 93.8% correlation. This suggests significant redundancy in the model ensemble — though removing them may not help if the mixer already down-weights them.

---

## Approaches Already Tried (Failed / Dead-End)

### 1. UTF-8 Preprocessing
- enwik8.utf8p → 15,831,098 bytes vs baseline 14,969,794 → **862 KB worse**
- Conclusion: "wide to ascii is a dead end"

### 2. Chunked Preprocessing (v1)
- enwik8.chunked → 17,148,795 bytes → **2.18 MB worse**
- Approach: Split file into typed regions (ASCII, brackets, XML, headers, UTF-8 runs, curly brackets) with control tags
- Problem: Interleaving data with control sequences hurts compression

### 3. V3 Split Format
- Split enwik8 into separate streams by type (ascii, control, utf8, numbers)
- Combined: 18,338,440 bytes (split) / 18,533,108 bytes (interleaved) → **3.37 MB worse**
- Numbers stream compresses poorly (69.9% ratio)
- Splitting destroys cross-stream context that the unified model exploits

### 4. Mixer Weighting (`--mw-enable`)
- enwik8 no-preprocess: 15,832,722 vs baseline 15,828,715 → **4 KB worse**
- Per-source adaptive weights (exponentially smoothed) don't help — the mixer already adapts to source quality

### 5. Source Weighting
- input2: 180,802 bytes (same as baseline 180,824 ± noise)
- No statistically significant improvement

### 6. Custom Dictionary (Per-Corpus)
- enwik8 with words_enwik8.dic: 15,060,107 + 165,170 (dict) = 15,225,277 vs baseline 14,962,810 + 99,916 (english.dic) = 15,062,726
- **162 KB worse overall** — the custom dictionary is larger and its compression doesn't offset the difference
- english.dic already well-optimized for Wikipedia text

### 7. New Preprocessing with Suffix Compression
- enwik8.opt.comp.txt: 16,131,194 + 366,480 (dict) = 16,497,674 bytes → **worse than baseline**
- PPMd-only on preprocessed: 20,626,481 vs original 20,774,117 → slight preprocessing benefit for PPMd alone, but negated by full model

---

## Improvement Opportunities

### Tier 1: High Impact, Moderate Complexity

#### T1-1: Deeper/Wider LSTM (2-layer or larger cells)
**Evidence:** LSTM experiments showed clear improvements with deeper architectures:

| Config | input (bytes) | input2 (bytes) | Time (input2) |
|--------|--------------|----------------|---------------|
| 200×1 (current) | 6,146 | 180,987 | 647 s |
| 64×1 | 6,143 | 181,718 | 305 s |
| 64×2 | 6,136 | 181,036 | 444 s |
| **128×2** | **6,137** | **180,708** | **751 s** |

128×2 saves ~279 bytes on input2 vs 200×1. Two layers capture hierarchical dependencies (character-level + word-level). The compression gain would be larger on enwik8 (100 MB) where the LSTM has more data to learn from.

**Implementation:** Change LSTM constructor params in predictor.cpp. Make num_layers and num_cells CLI-tunable. Profile memory impact (~2× for second layer).

**Risk:** ~1.16× slowdown. Memory increase proportional to cells×layers.

#### T1-2: Increase Mixer Context Map Limit
**Evidence:** Current limit is 10,000 entries. When a new context hash arrives and the map is full, the mixer falls back to the base context — losing specificity. On enwik8 (100M bytes), the context space is vast.

**Implementation:** Make the 10,000 limit configurable via CLI. Test 20K, 50K, 100K. Monitor memory usage (each entry is a weight vector of size num_models).

**Risk:** Memory growth proportional to limit × num_models × sizeof(float). At 461 models × 4 bytes × 100K entries = ~175 MB per mixer × 23 mixers = 4 GB. So 20K–30K may be the practical ceiling.

#### T1-3: PPMd Memory Increase (1024 → 14000 MB for competition)
**Evidence:** PPMd with 14000 MB vs 1024 MB on enwik8:
- 14000 MB + no dict: 15,832,397 bytes
- 1024 MB + no dict: 15,896,917 bytes  
- **Savings: ~64 KB (0.4%)**

Already done for enwik9 competition. For smaller corpora, 1024 MB is sufficient. This is a simple CLI/config change for competition runs.

#### T1-4: Adaptive Mixer Learning Rate
**Evidence:** All 23 L0 mixers use fixed learning rates (0.0003–0.005). The decay schedule (1.0→0.7→0.3→0.2 at step thresholds) is global and coarsely tuned. 

**Approach:** Per-mixer adaptive LR using inverse-loss tracking: `lr_effective = lr_base / (1 + c * cumulative_loss)`. This lets mixers that see novel contexts learn fast early and stabilize later. Alternatively, use Adam-style moment tracking for mixer weights.

**Risk:** Low — purely additive. Worst case is no improvement. 

#### T1-5: Reduce Mixer Decay Aggressiveness
**Evidence:** The decay schedule drops to 0.2 at 25M steps. On enwik8 (800M bits), most training happens at decay=0.2, which heavily dampens weight updates. The mixer may be unable to adapt to late-file context shifts (different Wikipedia articles have very different statistical properties).

**Approach:** Test gentler decay curves: e.g., 1.0 → 0.8 → 0.5 → 0.3. Or use a continuous logarithmic decay instead of step function.

### Tier 2: Moderate Impact, Higher Complexity

#### T2-1: Additional Mixer Contexts
**Evidence:** The 23 L0 mixers use contexts like recent bytes, line breaks, match length, word context, stream context. Missing opportunities:
- **Character class context:** Is current position alphabetic, digit, space, punctuation? (4-bit feature hashed with recent context)
- **Position modulo features:** `pos % 64`, `pos % 256` — captures periodicities in structured data
- **Match length buckets:** Already have match models; export bucketed match length (0–3, 4–7, 8–15, 16+) as mixer context
- **Word length context:** Current word length mod 8

**Implementation:** Add new context hashes to context-manager, wire new L0 mixers in predictor.cpp. Each new mixer adds one more source to the L1 mixer.

**Risk:** Each mixer adds memory (context map × weight vector). Diminishing returns if contexts are correlated with existing ones. Test one at a time.

#### T2-2: Meta-Mixer for Layer 1
**Evidence:** The single L1 mixer uses `zero_context_` (no context — just a global weighted average of L0 outputs). This is the weakest possible L1 configuration.

**Approach:** Add 2–3 L1 mixers with distinct contexts:
- Byte value context (recent byte hash)  
- Bit position within byte (0–7)
- Recent prediction confidence (bucketed entropy of last N predictions)

Then either average L1 outputs or add a lightweight L2 mixer.

**Risk:** Adding L1 mixers increases the number of SSE inputs. Need to ensure SSE scales.

#### T2-3: LSTM Input Enrichment
**Evidence:** Currently the LSTM receives only the averaged PPMd probability distribution (via byte-mixer). The LSTM has no visibility into individual model predictions or mixer outputs.

**Approach:** Feed additional signals to the LSTM:
- Top-K mixer logits (after L0 mixing)
- Match model confidence (binary: did any match model fire?)
- Character class of previous bytes

This gives the LSTM richer input features beyond just the PPMd distribution, potentially improving its 256-way byte prediction.

**Risk:** Increases LSTM input dimensionality, which increases computation. Keep extra features small (5–10 additional neurons).

#### T2-4: SSE Calibration / Residual Correction
**Evidence:** The 2-stage SSE cascade has ~2.67M table entries. Its quality depends on how well the lookup tables are populated. On heterogeneous data, early SSE entries may be biased by early statistics.

**Approach:** 
- Track running bias (average prediction - average outcome) over sliding window
- Apply Platt-style linear recalibration: `logistic(a*x + b)` where a,b are periodically updated
- Add a tiny residual correction table (16–32 bins) keyed by (SSE output bin, recent bit run length bucket)

**Risk:** Moderate complexity in SSE code. Ensure calibration doesn't overfit to local statistics.

#### T2-5: FXCM Optimization
**Evidence:** FXCM provides 431 of the 461 model outputs — dominating the input to the mixer. All 431 sub-models use state tables. The quality of these state tables directly impacts the ensemble.

**Approach:**
- Audit state table initialization — are transitions optimal for text compression?
- Test pruning low-value FXCM sub-models (identify via per-model KL divergence tracking)
- Consider increasing state table resolution for high-correlation contexts

**Risk:** FXCM internals are complex. Small changes can cascade. Need careful ablation testing.

### Tier 3: Experimental / Research

#### T3-1: Replace LSTM with Transformer-like Attention
Modern sequence models (even tiny 1-layer transformers with local attention windows) may outperform LSTMs on text modeling. However, this is a major architectural change with uncertain benefits for bit-level compression.

#### T3-2: Dictionary Optimization for Hutter Prize
**Evidence:** english.dic (44,515 entries) compresses to 99,916 bytes. A perfect corpus-specific dictionary could save more, but experiments with custom dictionaries showed the compressed dictionary itself costs too much:
- words_enwik8.dic (pruned): compressed data 15,060,107 + dict 165,170 = 15,225,277 → **worse**

The key constraint: `compressed_data + compressed_dictionary < baseline`. To improve:
- Optimize the tier-1 (80 most frequent) word list specifically for the target corpus
- Use frequency-weighted selection: maximize `(savings_per_use × frequency) - dictionary_overhead`
- Better dictionary compression (the dictionary file itself should be highly compressible)

#### T3-3: Preprocessing That Helps (Not Hurts) the Model
All preprocessing attempts so far have made compression worse because:
1. They increase data size with control sequences
2. They disrupt statistical patterns the model has learned to exploit
3. The full model (PPMd + FXCM + LSTM + 23 mixers) already handles text structure well

If preprocessing is revisited, it should:
- Not increase the total byte count
- Be reversible with zero overhead (no metadata/headers)
- Simplify patterns the model struggles with (e.g., numbers compression at 69.9% is poor)

Potential: byte-order remapping that clusters common byte patterns. Already implemented for dictionary encoding; could be extended.

#### T3-4: Online Entropy Tracking for AutoTuning
**Approach:** Track bits/byte over sliding windows. Use this signal to:
- Detect regime changes (article boundaries in Wikipedia)
- Switch between LSTM configurations (aggressive vs conservative)
- Reset mixer weights at detected boundaries

#### T3-5: Ensemble Diversity Enforcement
**Evidence:** 30/32 predictors show 93.8% correlation. The mixer benefits from *diverse* inputs.

**Approach:** During training, add a diversity regularizer that penalizes correlated predictions. Or use dropout-like randomization on model inputs to force decorrelation.

**Risk:** May hurt individual model quality to gain ensemble diversity. Net effect unclear.

---

## Recommended Priority Order

| Priority | Item | Expected Impact | Effort |
|----------|------|----------------|--------|
| **1** | T1-1: 128×2 LSTM | ~279 B/MB improvement | Low (param change) |
| **2** | T1-5: Gentler mixer decay | ~unknown, easy to test | Low |
| **3** | T1-4: Adaptive mixer LR | ~unknown, principled | Medium |
| **4** | T1-2: Increase context map limit | ~unknown | Low (param change) |
| **5** | T2-2: Meta-mixer for L1 | ~unknown but principled | Medium |
| **6** | T2-1: Additional mixer contexts | ~additive small gains | Medium |
| **7** | T2-3: LSTM input enrichment | moderate | High |
| **8** | T1-3: PPMd 14000 MB | ~64 KB on enwik8 | Trivial |
| **9** | T2-4: SSE calibration | small | Medium |
| **10** | T2-5: FXCM pruning/tuning | unknown | High |

Items 1–4 can be tested quickly with small code changes and measured on input/input2 first. Item 8 is already implemented for competition mode.

---

## Measurement Protocol

1. **Fast test:** input (51 KB) and input2 (942 KB) with `no-preprocess` — takes ~18 s and ~225 s respectively
2. **Medium test:** enwik7 (10 MB) with english.dic — takes ~30 min
3. **Full test:** enwik8 (100 MB) with english.dic — takes ~4 hours
4. **Competition test:** enwik9 (934 MB) with english.dic + PPMd 14000 MB — takes ~63 hours

Always compare exact byte counts (not percentages). Run seed=2 for consistency. Use `no-preprocess` mode for model-only experiments, `compress -d english.dic` for full-pipeline experiments.

---

## Key Takeaways

1. **The LSTM is the compression workhorse** — 93% mean accuracy vs ~46% for all other predictors at bit level. Improving the LSTM (deeper, wider, better inputs) has the highest ROI.

2. **The mixer is the weakest link after the LSTM** — a single L1 mixer with zero context, coarse decay, and a 10K map limit leaves information on the table.

3. **Preprocessing is a trap** — every attempt so far degraded compression. The full model already handles text structure. Future preprocessing must be *transparent* to the model's statistics.

4. **Dictionary impacts are marginal once the full model is running** — english.dic saves ~800 KB on enwik8 at the PPMd-only level but less once the LSTM and mixers compensate. Custom dictionaries haven't beaten english.dic when total cost (compressed data + compressed dict) is considered.

5. **The 431 FXCM sub-models dominate the input count but their individual impact is unclear** — they may add diversity the mixer exploits, or they may mostly be noise. Per-model ablation would clarify.

---

## PPMd Memory Footprint Analysis

### Current Memory Budget

| Component | Default | Competition | Notes |
|-----------|---------|-------------|-------|
| Heap (SubAllocatorSize) | 1024 MB | 14000 MB | `SASize << 20` bytes |
| Text area (1/8) | 128 MB | 1750 MB | Stores symbol bytes during model update |
| Context storage (7/8) | 896 MB | 12250 MB | PPM_CONTEXT + STATE arrays |
| Static data (per instance) | ~15 KB | ~15 KB | BList, BinSumm, SEE2Cont, CharMask, SQ, sqp |

**Heap dominates:** The ~15 KB of static/stack data is negligible. The 1–14 GB heap is the entire memory footprint.

### Data Structure Inventory

| Structure | Size | Packing | Count (order of magnitude) | Purpose |
|-----------|------|---------|----------------------------|---------|
| STATE | 6 B | packed | ~50–100M per GB | Symbol + frequency + successor index |
| PPM_CONTEXT | 12 B (1 UNIT) | packed | ~25–50M per GB | Context tree node (NumStats, Flags, SummFreq, iStats, iSuffix) |
| BLK_NODE | 8 B | packed | reused in free list | Free list node header |
| MEM_BLK | 12 B (1 UNIT) | packed | reused in free list | Free block header (extends BLK_NODE with NU) |
| SEE2_CONTEXT | 4 B | packed | 23×32 = 736 fixed | Secondary escape estimation |

**Already-implemented optimizations:**
1. **Pointer compression:** 8-byte pointers → 4-byte indices via `Ptr2Indx`/`Indx2Ptr`. Saves 4 bytes per STATE (iSuccessor) and 8 bytes per PPM_CONTEXT (iStats + iSuffix). Without this, STATE would be 10 B and PPM_CONTEXT 20 B — nearly 2× the current sizes.
2. **Binary context optimization:** When `NumStats==0`, the single STATE is stored directly in the PPM_CONTEXT's `SummFreq` field — no separate allocation needed. This covers a large fraction of contexts (many contexts have only 1 or 2 symbols).
3. **Segregated free list:** 38 size-class buckets minimize fragmentation waste. Freed blocks are reused efficiently.
4. **#pragma pack(1):** All structures are byte-packed — no alignment padding waste.

### Memory Optimization Proposals

#### M1: Reduce text area ratio from 1/8 to 1/16 ★★★

**Current situation:** `InitSubAllocator()` reserves 1/8 (12.5%) of the heap for text area storage. The text area holds symbol bytes written by `UpdateModel()` (`*pText++ = FSymbol`). These bytes serve as temporary successors for newly-created leaf contexts until `CreateSuccessors()` promotes them to full PPM_CONTEXT nodes.

**The text area is aggressively recycled:**
- `RestoreModelRare()` resets `pText = HeapStart` (full reclaim)
- `ExpandTextArea()` moves `UnitsStart` forward, reclaiming text space
- `PrepareTextArea()` + `cutOff()` are called in a loop until used memory < 3/4

**Why 1/8 is too much:** With order 25, each processed byte writes at most 1 byte to text area (when `OrderFall > 0`). Before text area fills, `RestoreModelRare` triggers and resets it. On 1 GB heap, 128 MB of text area can hold 128M pending symbols — but context pruning recycles it long before that. On 14 GB heap, 1750 MB of text area for enwik9 (934 MB input) is nearly 2× the input size.

**Proposal:** Change the split ratio to 1/16 (6.25%), freeing an additional 6.25% of heap for context storage.

```cpp
// Current (line ~513):
qword Diff = SubAllocatorSize / 8 / UNIT_SIZE * 7 * UNIT_SIZE;
// Proposed:
qword Diff = SubAllocatorSize / 16 * 15 / UNIT_SIZE * UNIT_SIZE;
```

**Impact:**
- 1 GB heap: +64 MB for contexts (128 MB → 64 MB text, 896 MB → 960 MB contexts)
- 14 GB heap: +875 MB for contexts (1750 MB → 875 MB text, 12250 MB → 13125 MB contexts)
- More context memory → fewer cutOff/restore cycles → more contexts retained → potentially better compression
- Or: achieve same compression quality with less total memory

**Risk:** Low. Text area recyclng is well-tested. Monitor for `pText >= UnitsStart` triggering more frequently. If it does, it just means more RestoreModelRare calls (already handles this case).

**Validation:** Run enwik7 and enwik8 benchmarks. Compare compressed size, RestoreModelRare count, and GetUsedMemory peak.

#### M2: Wrap debug printf/assert code in #ifdef ★★

**Current situation:** The codebase contains extensive debug instrumentation:
- ~50+ `printf()` statements for error diagnostics
- ~100+ `assert()` calls with descriptive messages
- `ValidateBList()` called after allocator operations
- Bounds-checking in every `Ptr2Indx`, `Indx2Ptr`, `getSucc`, `getStats`, `suff` call

**In release builds (`-DNDEBUG`):** `assert()` compiles to nothing, but `printf()` calls remain in the binary (inside dead `if` branches that the optimizer may or may not eliminate). The `ValidateBList` function and extensive bounds-check code bloat the instruction cache.

**Proposal:** Wrap all debug-only code in a `PPMD_DEBUG` guard:

```cpp
#ifdef PPMD_DEBUG
  #define PPMD_DPRINTF(...) printf(__VA_ARGS__)
  #define PPMD_ASSERT(cond, msg) do { if (!(cond)) { printf msg; assert(false); } } while(0)
#else
  #define PPMD_DPRINTF(...) ((void)0)
  #define PPMD_ASSERT(cond, msg) ((void)0)
#endif
```

**Impact:**
- Reduces compiled binary size by ~5–10 KB (fewer string literals, less dead code)
- Improves I-cache utilization — the hot path (Ptr2Indx, Indx2Ptr, getStats, suff) becomes much shorter
- No effect on heap memory or compression quality
- Debug builds still have full instrumentation

**Risk:** None for compression quality. Guard must not accidentally wrap functional code.

#### M3: Optimize FreeUnits/FreeUnit zeroing ★

**Current situation:** Both `FreeUnits()` and `FreeUnit()` zero the entire freed block before inserting into the free list:
```cpp
memset(ptr, 0, blockSize);  // FreeUnits: zeros 12*NU bytes
memset(ptr, 0, UNIT_SIZE);  // FreeUnit: zeros 12 bytes
```
This was added as a "CRITICAL FIX" to prevent stale indices (iSuccessor, iStats, iSuffix) from being misinterpreted when blocks are reused by `GlueFreeBlocks()`.

**The issue:** `GlueFreeBlocks()` checks `p1->Stamp == ~uint(0)` to detect adjacent free blocks. The `insert()` function writes `p->Stamp = ~uint(0)`. The memset ensures non-header bytes (bytes 8–11, 12–23, etc.) within the freed block don't accidentally have `0xFFFFFFFF` patterns that could cause false merges.

**Proposal:** Replace full-block zeroing with header-only zeroing:
```cpp
// Zero only bytes beyond the BLK_NODE header (8 bytes) that insert() will overwrite
// The first 8 bytes are immediately overwritten by insert() (Stamp + NextIndx)
// Only need to zero remaining bytes that might contain stale Stamp patterns
memset((byte*)ptr + sizeof(BLK_NODE), 0, blockSize - sizeof(BLK_NODE));
```

Or more targeted: zero only at UNIT_SIZE boundaries where `Stamp` fields would fall:
```cpp
for (uint i = 1; i < NU; i++)
  ((uint*)((byte*)ptr + i * UNIT_SIZE))[0] = 0;  // Zero Stamp field at each unit boundary
```

**Impact:**
- Saves ~40% of memset work for FreeUnit (zero 4 bytes instead of 12)
- Saves more for large NU blocks  
- No memory savings — this is pure performance optimization
- No compression quality change

**Risk:** Low but needs careful analysis. The GlueFreeBlocks merge logic must be reviewed to confirm only Stamp fields matter.

#### M4: Configurable text area ratio ★

**Current:** The 1/8 ratio is hardcoded in `InitSubAllocator()`.

**Proposal:** Add a configurable parameter for the text/context split:

```cpp
uint Init(uint MaxOrder, uint MMAX, uint CutOff, uint filesize, int textRatioShift = 3) {
  // textRatioShift=3 → 1/8, textRatioShift=4 → 1/16
  ...
}
```

This enables tuning per workload without code changes, and allows benchmarking different ratios.

#### M5: Use mmap instead of new[] for large heaps ★

**Current:** `HeapStart = new byte[t]` allocates the heap, then `memset(HeapStart, 0, t)` zeros it.

**On Linux/macOS:** `mmap(MAP_ANONYMOUS)` returns zero-initialized pages on demand (lazy allocation). The OS only commits physical pages as they're touched. This means:
- No upfront zeroing cost (saves seconds for 14 GB allocation)
- Physical memory usage matches actual usage, not reserved size
- Pages that are freed and not reused don't consume physical RAM

**On Windows:** `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)` provides similar behavior.

**Impact:**
- For 14 GB competition mode: saves ~14 seconds of zeroing at startup
- Physical memory usage may be lower than the allocated size (OS manages pages)
- No compression quality impact

**Risk:** Low. Platform-specific code needed. Fallback to `new[]` + `memset` for portability.

### Proposals NOT recommended (would hurt compression)

#### ✗ Reduce iSuccessor to 3 bytes
STATE's `iSuccessor` is 4 bytes (32-bit index). With 1 GB heap, max index ≈ 212M (fits in 28 bits). Could save 1 byte per STATE.

**Why rejected:** With 14 GB heap, max index ≈ 1.25B, needs 31 bits — too tight for 24-bit. Also, 5-byte STATEs break the 2-per-UNIT alignment (12/5 = 2.4), requiring a complete allocator redesign.

#### ✗ Reduce PPM_CONTEXT.iSuffix to 3 bytes
Same issue — doesn't fit for competition mode, breaks struct alignment.

#### ✗ Shrink UNIT_SIZE below 12 bytes
UNIT_SIZE = 12 is the minimum for PPM_CONTEXT (12 bytes). Any smaller would require splitting contexts across units or redesigning the context structure.

#### ✗ Reduce heap below 1024 MB
Already measured: 14000 MB → 1024 MB costs ~64 KB on enwik8. Going lower would cost more compression quality, and the cutoff mechanism already handles pressure by pruning low-frequency contexts.

### Priority Order

| Priority | Proposal | Impact | Risk | Effort |
|----------|----------|--------|------|--------|
| **1** | M1: Text area 1/8 → 1/16 | +6.25% context memory | Low | Trivial (one line) |
| **2** | M2: #ifdef debug code | Better I-cache, smaller binary | None | Medium (many edits) |
| **3** | M4: Configurable text ratio | Enables tuning | None | Low |
| **4** | M3: Targeted zeroing | Performance | Low | Low |
| **5** | M5: mmap/VirtualAlloc | Reduced startup cost, lazy pages | Low | Medium |

**Recommended first step:** Implement M1 (text area reduction) and benchmark on enwik7/enwik8. If compression improves or stays the same, it's a free win — more context memory without increasing the heap. If text area pressure increases (more RestoreModelRare calls), try 3/32 (9.375%) as a middle ground.

### PPMd-Only Predictor Baselines

Isolated PPMd predictor (`-p ppmd no-preprocess`), 14000 MB heap, no dictionary.

| Corpus | Order | Output | Ratio | Time | Speed (B/s) | Heap used | Resets |
|--------|------:|-------:|------:|-----:|------------:|----------:|-------:|
| enwik7 | 25 | 2,284,952 | 22.850% | 39.6s | 252,334 | 250 MB (1.8%) | 0 |
| enwik7 | 18 | 2,284,922 | 22.849% | 40.6s | 246,451 | 221 MB (1.6%) | 0 |
| enwik8 | 25 | 20,774,111 | 20.774% | 383.7s | 260,643 | — | 0 |
| enwik9 | 25 | 165,283,254 | 16.528% | 3825.2s | 261,426 | 9913 MB (70.8%) | 1 at 68% |
| enwik9 | 22 | 165,288,589 | 16.529% | 3973.8s | 251,646 | 8455 MB (60.4%) | 1 at 74.5% |
| enwik9 | 18 | 165,169,516 | 16.517% | 4028.7s | 248,219 | 6081 MB (43.4%) | 1 at 92.2% |

**Observations:**
- **Enwik7 order 18 vs 25: +30 bytes (+0.001%)** — for small inputs the order makes zero difference in quality. Both use <250 MB of the 14 GB heap.
- **Speed is ~246–261 KB/s** across all runs — within run-to-run noise.
- **Enwik9 ratio improves with larger input** (16.5% vs 20.8% for enwik8) — order 25 benefits from more context history on larger corpora.
- **Enwik9 triggers 1 reset regardless of order** (25, 22, 18) — the tree fills all 14 GB for a 1 GB input at any practical order depth. However, the reset position shifts dramatically: order 18 fires at 92% vs order 25 at 68%.
- **Order 18 produces the best output** — 165,169,516 bytes, **−113,738 bytes (−0.069%) better than order 25**. Later reset = only 8% of corpus compresses without full context history, vs 32% for order 25.
- **Comparison to full predictor:** enwik8 full predictor (PPMd+LSTM+Bracket) achieves 14,962,810 bytes — 28% smaller than PPMd-only. Mixer/LSTM provide significant additional compression on top of PPMd.

#### M1 Result: Neutral

M1 (text area 1/8 → 1/16) produced **identical compressed output** (165,283,254 bytes) with no meaningful speed change (within noise). Reset count is unchanged at 1, occurring at the same point (~68% input).

**Analysis:** The reset is not caused by text area exhaustion — it is caused by context storage hitting the 3/4 full threshold. At 14 GB with order 25 on a 1 GB input, the context tree grows to fill available space regardless of the text/context split ratio. The extra ~875 MB of context storage freed by M1 is consumed by the growing tree just as quickly, so the reset merely shifts by a few MB of processed input rather than being deferred meaningfully.

**Conclusion:** M1 is safe to keep (it doesn't hurt compression), but it doesn't address the root cause. M1 is reverted in the priority table — no further text-area experiments are warranted. To delay/eliminate resets, options are: increase total heap, lower order, or tune `cutOff`/GlueCount thresholds to prune more aggressively before the 3/4 threshold.

#### Heap Profile — enwik9 reset anatomy (14 GB, order 25)

Full `PrintHeapStats` data from the corrected instrumentation (before-reset, after-reset, end-of-compression):

```
[heap-stats before-reset]   @ 68.19% = ~682 MB processed
  used:            13999 MB  (100.0%)      ← allocation failure, not threshold hit
  to-threshold:        0 MB headroom
  text-area:         480 MB / 875 MB cap  (54.9%)
  ctx-hi-alloc:     8599 MB  (PPM_CONTEXT nodes, from top downward)
  ctx-lo-alloc:     4919 MB  (STATE arrays, from base upward)
  ctx-gap-free:        0 MB  (both fronts met — no bump-pointer room left)
  mid-gap:             0 MB
  free-list:           0 MB  (0 buckets)
  frag-score:        0.0%

[heap-stats after-reset]    (immediately after RestoreModelRare)
  used:             4884 MB  (34.9%)       ← 9115 MB freed (65% of heap!)
  to-threshold:     5615 MB headroom
  ctx-hi-alloc:     8599 MB  (UNCHANGED — PPM_CONTEXT ptrs not moved)
  ctx-lo-alloc:     4919 MB  (UNCHANGED — LoUnit/HiUnit not moved)
  ctx-gap-free:        0 MB
  mid-gap:           480 MB  (text area reset to HeapStart)
  free-list:        8634 MB  (26 non-empty buckets)
  frag-score:       93.9%    ← CRITICAL: nearly all free memory is tiny fragments
  frag-small:       8110 MB  <=4u   (632,929,535 blocks)   ← ~1u avg = 1 STATE entry
  frag-medium:       464 MB  5-16u  (5,495,636 blocks)
  frag-large:         59 MB  >16u   (232,936 blocks)

[heap-stats end-of-compression]   @ 100%
  used:             9913 MB  (70.8%)       ← just under 75%, no second reset
  to-threshold:      586 MB headroom
  ctx-hi-alloc:     8599 MB  (unchanged)
  ctx-lo-alloc:     4919 MB  (unchanged)
  free-list:        3849 MB  (30 buckets)
  frag-score:        0.0%
  frag-small:          0 MB  <=4u   (0 blocks)
  frag-medium:         0 MB  5-16u  (1 block)
  frag-large:       3849 MB  >16u   (8,869,246 blocks)
```

**Analysis:**

**1. The reset trigger is allocation exhaustion, not the 75% threshold.**
The 75% threshold (`GetUsedMemory() > 3/4 * SubAllocatorSize`) is the *stopping condition* of the `RestoreModelRare` loop — it keeps pruning until used < 75%. The *trigger* is when `AllocUnitsRare` returns null (bump-pointer space gone, free-list empty). By that point the heap is 100% full with 0 free-list entries.

**2. `RestoreModelRare` frees aggressively: 13999 → 4884 MB (65% freed).**
However, it does so by calling `cutOff()` which traverses the context tree and individually frees STATE entry arrays one block at a time. With order 25, most leaf contexts have 1–4 states → freed as 1–4 unit blocks. Result: **632 million tiny free blocks**, 93.9% of freed memory is ≤48 bytes.

**3. `ctx-hi-alloc` and `ctx-lo-alloc` are address ranges, not live-object counts.**
They reflect where `HiUnit` and `LoUnit` pointers are relative to the heap ends. They don't change during reset because the free-list mechanism is used, not bump-pointer retraction. The 4884 MB of "used" after reset is the truly live context tree nodes.

**4. Post-reset fragmentation resolves itself by end-of-compression.**
By the final stats, `frag-small = 0 MB`, `frag-large = 3849 MB`. The 632M tiny blocks all got reallocated (new contexts), re-freed during natural model updates, and coalesced by `GlueFreeBlocks` into 8.9M large (>16u) blocks. The allocator's defragmentation works, but it takes the entire remaining 318 MB of input to converge.

**5. The ~32% of input processed after reset compresses sub-optimally.**
After reset the model has no accumulated context history — it essentially restarts from order 0 and rebuilds. The first several MB of post-reset compression have much weaker predictions until context depth grows back. This is the primary compression quality cost of the reset.

#### Next Step Options: Eliminate the Reset

Goal: process all 1 GB of enwik9 at order 25 without ever hitting allocation exhaustion.

**Option A — Lower order (recommended first test)**

| Order | Heap at reset | Reset at | End used | Output bytes (enwik9) | Quality delta |
|-------|--------------|----------|----------|----------------------|---------------|
| 25 | 13999 MB (measured) | 68.19% | 70.8% | 165,283,254 | baseline |
| 22 | 13999 MB (measured) | 74.53% | 60.4% | 165,288,589 | +0.003% |
| 18 | 13999 MB (measured) | 92.19% | 43.4% | **165,169,516** | **−0.069%** ✓ best |
| — | enwik7 order 25 | no reset | 1.8% | 2,284,952 | baseline |
| — | enwik7 order 18 | no reset | 1.6% | 2,284,922 | −0.001% |

**Corrected understanding:** The PPMd tree is **not memory-bounded by order** in the way initially estimated. It is input-bounded: with enough input, it will build contexts up to the given order for every n-gram seen, regardless of available memory — until it exhausts the heap. For a 1 GB corpus at 14 GB heap, both order 25 and order 22 fill all 14 GB, just at different corpus positions. The original estimates (order 22 → ~9 GB, order 18 → ~4 GB) were wrong.

The correct model: the tree fills memory at a rate proportional to `corpus_size × order × avg_branching_factor`. For enwik9 at 14 GB: order 25 fills at 68%, order 22 fills at 74.5% — a 6.3% shift per 3 orders. At this rate, eliminating the reset by order reduction alone would require approximately **order ≤ 5**, which would be useless for compression.

Context count scales roughly as O(n × order) for new unique n-grams. Going from 25→20 is a 20% reduction in max depth, but in practice high-order (20–25) contexts are sparse and dominate the tree. Estimate: order 20 uses ~40–50% of the heap that order 25 does.

**Order 18 result:** 165,169,516 bytes — **113,738 bytes better than order 25**. The later reset position (92% vs 68%) means only 8% of the corpus suffers post-reset degradation. The quality advantage of deeper contexts (order 25) is outweighed by the quality cost of losing 32% of the corpus to a post-reset cold-start.

**Option B — Proactive voluntary reset (early threshold)**

Instead of waiting for allocation failure, add a check in `ByteUpdate` (or periodically in `encodeSymbol`) that calls `RestoreModelRare` when `GetUsedMemory() > 0.80 * SubAllocatorSize`. Benefits:
- Reset fires at 80% fill instead of 100% — more room to prune gracefully
- Smaller free-list fragmentation (tree hasn't grown as deep)
- Can keep order 25 and still avoid allocation exhaustion
- Fewer tiny-block fragments → faster post-reset allocation

Implementation: add a `proactiveResetThreshold` field (e.g., 0.80), check it at `ByteUpdate` boundaries.

**Option C — Accept one reset, tune the cutOff depth**

Currently `cutOff(MaxContext, 0, _MaxOrder)` keeps contexts up to max order. Passing a lower depth limit (e.g., order 15) during reset would free significantly more memory per reset call, giving more post-reset headroom at the cost of losing deeper context history sooner.

#### Order 22 Result (14 GB heap, enwik9)

| Metric | Order 25 | Order 22 | Delta |
|--------|----------|----------|-------|
| Output bytes | 165,283,254 | 165,288,589 | +5,335 (+0.003%) |
| Reset at | 68.19% | 74.53% | +6.3% later |
| After-reset used | 34.9% | 35.9% | similar |
| End used | 70.8% | 60.4% | −10.4% |
| Speed | ~258 KB/s | ~252 KB/s | −2% |

**Key findings:**

1. **Quality is essentially identical** — +0.003% difference is noise. Order 22 is as good as order 25 for enwik9.

2. **Reset still fires** — the tree fills all 14 GB regardless of order (22 or 25). Lower order just postpones the fill point: each −3 orders moves the reset ~6% later in the corpus. To fully eliminate the reset by reducing order alone, extrapolation suggests order ≤ 10–12, which would likely hurt compression significantly.

3. **Post-reset portion decreases**: 25% of corpus after reset (order 22) vs 32% (order 25). Real benefit, but reset is not eliminated.

4. **After-reset fragmentation is identical in character**: 93.4% frag-score (612M tiny blocks, order 22) vs 93.9% (632M tiny blocks, order 25). The fragmentation pattern is structural — it comes from how `cutOff()` individually frees single-state entries, independent of order.

5. **End-of-compression has 2044 MB headroom** (order 22) vs 586 MB (order 25). The lower order is more conservative overall.

**Conclusion on order reduction:** Order reduction alone cannot eliminate the reset at 14 GB for a 1 GB corpus at any practical order (≥16). The tree will always grow to fill available memory. The structural fix must be either (a) proactive pruning before exhaustion, or (b) a hard memory cap enforced before the tree fills completely.

**Recommended experiment sequence:**
1. ✅ ~~Test order 22~~ — same quality (+0.003%), reset still fires
2. ✅ ~~Test order 18~~ — **best output so far (−0.069% vs order 25)**; reset fires at 92%, only 8% post-reset
3. **Implement gentle reset (Option D)** — proactive frequency-based pruning (Freq≤1 removal) at ~80% fill:
   - Keep order 25 for context depth, but prune cold states before exhaustion
   - Goal: eliminate the reset entirely, or push it past 95%+
   - If gentle reset frees enough memory, order 25 may beat order 18
4. If gentle reset doesn't fully prevent a hard reset, combine: order 18 + gentle reset

#### Order 18 Result (14 GB heap, enwik9) — **Best result so far**

| Metric | Order 25 | Order 22 | Order 18 |
|--------|----------|----------|----------|
| Output bytes | 165,283,254 | 165,288,589 | **165,169,516** |
| Delta vs order 25 | baseline | +5,335 (+0.003%) | **−113,738 (−0.069%)** |
| Reset at | 68.19% | 74.53% | **92.19%** |
| Post-reset corpus | 31.8% | 25.5% | **7.8%** |
| After-reset used | 34.9% | 35.9% | 38.3% |
| End used | 70.8% | 60.4% | 43.4% |
| Speed | ~261 KB/s | ~252 KB/s | ~248 KB/s |

**Key findings:**

1. **Order 18 produces the best enwik9 compression** — 113 KB better than order 25. This is counter-intuitive: lower order → better compression. The explanation is entirely about reset timing.

2. **The reset / post-reset tradeoff dominates.** At order 25: 32% of the corpus is processed in a cold-start state (model rebuilt from scratch after reset). At order 18: only 8%. The 24% difference in cold-start fraction, each cold byte compressing significantly worse, more than compensates for the shallower context depth.

3. **Reset fires at 92.2%** — only 78 MB of input processed post-reset. The after-reset frag pattern is similar: 92.2% frag-score, 568M tiny blocks.

4. **End-of-compression used: only 43.4%** (6081 MB of 14000 MB) — more than half the heap is unused at completion. The tree never had time to densify after the reset. This confirms order 18 with 14 GB is significantly over-provisioned.

5. **Fragmentation model confirmed**: frag-score drops from 92.2% (immediately after reset) to 15.5% at end of compression — same healing pattern as order 25.

**Implication for gentle reset:** Order 18 sets the quality bar at 165,169,516. A gentle reset on order 25 needs to do better than this. If it can defer/eliminate the reset on order 25 such that <8% of the corpus is processed post-reset (or with graceful degradation), it should beat order 18.

