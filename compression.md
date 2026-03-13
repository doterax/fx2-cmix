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
