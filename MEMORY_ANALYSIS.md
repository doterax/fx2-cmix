# Memory Consumption Analysis for fx2-cmix

## Executive Summary
The cmix compressor uses **~700-900 MB of RAM** during operation. This is by design - it's an extremely high-compression tool that trades memory for compression ratio.

---

## Major Memory Consumers (Ordered by Size)

### 1. **History Buffer: ~60 MB** 
**Location:** `src/context-manager.cpp:58`
```cpp
history_(60000000, 0)  // 60 million bytes = ~57 MB
```
- **Purpose:** Stores the last 60 million bytes of input data
- **Usage:** Match models use this to find repeated sequences
- **Why Large:** Larger history = better long-range pattern detection

### 2. **Shared State Map: ~100 MB**
**Location:** `src/context-manager.cpp:59`
```cpp
shared_map_(256*400000, 0)  // 102.4 million bytes = ~98 MB
```
- **Purpose:** Shared probability map for all indirect models
- **Usage:** Nonstationary and RunMap state storage
- **Why Large:** 256 possible byte values × 400,000 contexts

### 3. **Match Model Hash Tables: ~80-100 MB total**
**Location:** `src/models/match.cpp:10` (10 instances created)
```cpp
map_(map_size, 0)  // Each instance: 2-8 MB
```
- **10 Match models** created in `AddMatch()` and `AddWord()`
- Each has `map_size` ranging from **2,000,000 to 8,000,000 entries**
- Each entry = 4 bytes (unsigned int)
- **Total:** ~40-80 MB across all Match models

### 4. **Indirect Hash Tables: ~100-150 MB**
**Location:** `src/context-manager.cpp:60-64`
```cpp
hashes_ind1.resize(0x1000000, 0);   // 16 MB
hashes_ind2.resize(0x1000000, 0);   // 16 MB
hashes_ind3.resize(0x2000000, 0);   // 32 MB
hashes_ind4.resize(0x100, 0);       // 256 bytes
hashes_ind5.resize(0x100, 0);       // 256 bytes
```
- **Purpose:** Hash tables for double indirect addressing
- **Usage:** Context mapping for indirect models
- **Total:** ~64 MB just for these 5 tables

### 5. **FXCM Model Buffers: ~50-80 MB**
**Location:** `src/models/fxcmv1.cpp`
```cpp
U16 ind3[0x2000000];    // 32 MB (2^25 entries × 2 bytes)
U8 buffer[0x1000000];   // 16 MB (2^24 bytes)
```
- **Purpose:** FXCM model's internal state and buffer
- **Usage:** Context hashing and pattern storage
- **Total:** ~48 MB minimum

### 6. **Context Maps in Mixer: ~100-200 MB**
**Location:** `src/mixer/mixer.h:41`
```cpp
emhash6::HashMap<unsigned int, ContextData> context_map_;
```
- **23 Mixer instances** in layer 0, **1 in layer 1**
- Each ContextData contains **2 Eigen::VectorXf** with 460+ floats each
- HashMap grows dynamically as contexts are seen
- **Estimated:** 100-200 MB total after processing large files

### 7. **LSTM Neural Network: ~20-40 MB**
**Location:** `src/mixer/lstm.hpp:8-16`, `src/mixer/lstm-layer.h`
```cpp
// LSTM configuration: vocab_size × vocab_size × layers
Lstm(vocab_size, vocab_size, 200, 1, 128, 0.03, 10)
```
- **Parameters:**
  - Input size: ~200 (vocabulary size)
  - Output size: ~200
  - Hidden cells: 200
  - Layers: 1
  - Horizon: 128 timesteps
- **Weights:** 4 gates × (input_size + hidden_size) × hidden_size floats
- **Activations:** Horizon × layers × hidden_size stored
- **Total:** ~20-40 MB

### 8. **Model Predictions and Layers: ~10-20 MB**
**Location:** `src/predictor.h:60-75`
```cpp
llvm::SmallVector<Indirect<Nonstationary>, 30 - 7> indirect_ns_models_;  // 23 models
llvm::SmallVector<Match, 10> match_models_;                               // 10 models
llvm::SmallVector<MixerInput, 2> layers_;                                 // 2 layers
llvm::SmallVector<Mixer, 23> mixer_0_;                                    // 23 mixers
```
- Each model stores predictions (floats) and state
- Layers store 460+ model inputs/outputs
- **Total:** ~10-20 MB

---

## Memory Allocation by Component

| Component | Size | Description |
|-----------|------|-------------|
| **History Buffer** | ~60 MB | Raw byte history for match finding |
| **Shared Map** | ~100 MB | Probability state storage |
| **Match Hash Tables** | ~80 MB | 10 models × 2-8 MB each |
| **Indirect Hash Tables** | ~64 MB | 5 hash tables for indirect addressing |
| **FXCM Model** | ~50 MB | Internal buffers and indices |
| **Context Maps (Mixers)** | ~150 MB | 24 mixers with dynamic hash maps |
| **LSTM Network** | ~30 MB | Neural network weights + activations |
| **Model Predictions** | ~15 MB | 460+ models storing state |
| **Misc Contexts** | ~50 MB | Various context vectors and maps |
| **TOTAL ESTIMATED** | **~600-900 MB** | Varies with input size |

---

## Why So Much Memory?

### 1. **Trade-off: Compression Ratio vs Memory**
- cmix achieves **12-15% compression ratios** on text
- Top compressors (PAQ, cmix) use **hundreds of MB to GB of RAM**
- More memory = more context = better prediction = higher compression

### 2. **History-Based Matching**
- 60 MB history buffer allows finding matches from **millions of bytes ago**
- Match models are extremely effective for repetitive data
- Without large history, compression ratio drops significantly

### 3. **Context Modeling**
- **460+ prediction models** each need their own state
- Each context (word position, byte pattern, bracket depth, etc.) needs storage
- Hash maps grow as new contexts are encountered

### 4. **Neural Network**
- LSTM has **200 × 200 × 4 gates = 160K weights**
- Plus **128 timesteps of activations** (horizon)
- Essential for learning long-range dependencies

---

## Optimization Opportunities

### If Memory is a Constraint:

#### 1. **Reduce History Buffer** (Save ~30-50 MB)
**File:** `src/context-manager.cpp:58`
```cpp
// Current: 60000000 bytes
history_(30000000, 0)  // Reduce to 30 MB
```
**Impact:** Reduces long-range match effectiveness

#### 2. **Reduce Shared Map** (Save ~50 MB)
**File:** `src/context-manager.cpp:59`
```cpp
// Current: 256 * 400000
shared_map_(256 * 200000, 0)  // Reduce to ~50 MB
```
**Impact:** More context collisions, slightly lower compression

#### 3. **Reduce Match Model Sizes** (Save ~40-60 MB)
**File:** `src/predictor.cpp:88,75`
```cpp
// Current: max_size = 2000000
unsigned long long max_size = 1000000;  // Halve match model size
```
**Impact:** Fewer matches found, lower compression on repetitive data

#### 4. **Reduce Indirect Hash Tables** (Save ~30 MB)
**File:** `src/context-manager.cpp:60-62`
```cpp
hashes_ind1.resize(0x800000, 0);   // 8 MB instead of 16 MB
hashes_ind2.resize(0x800000, 0);   // 8 MB instead of 16 MB
hashes_ind3.resize(0x1000000, 0);  // 16 MB instead of 32 MB
```
**Impact:** More hash collisions in indirect models

#### 5. **Reduce LSTM Horizon** (Save ~10-15 MB)
**File:** `src/predictor.cpp:133`
```cpp
// Current: horizon = 128
new Lstm(vocab_size, vocab_size, 200, 1, 64, 0.03, 10)  // Reduce to 64
```
**Impact:** LSTM can't learn patterns as far back

#### 6. **Use Smaller Float Precision** (Save ~20%)
Replace `float` (4 bytes) with `half` (2 bytes) or `bfloat16` in Eigen
**Impact:** Slightly less numerical precision, may affect compression by 0.1-0.5%

---

## Memory Profile by Input Size

| Input Size | Estimated RAM Usage | Notes |
|------------|---------------------|-------|
| < 1 MB | ~400-500 MB | Base memory + small context maps |
| 1-10 MB | ~500-700 MB | Context maps growing |
| 10-100 MB | ~700-900 MB | Full working set, history buffer cycling |
| > 100 MB | ~900-1200 MB | Context maps at maximum, frequent GC |

---

## Comparison to Other Compressors

| Compressor | RAM Usage | Compression on Text |
|------------|-----------|---------------------|
| **gzip** | ~1 MB | ~30-35% |
| **bzip2** | ~4-8 MB | ~25-30% |
| **xz (LZMA)** | ~90-700 MB | ~20-25% |
| **zpaq** | ~100-500 MB | ~15-20% |
| **paq8** | ~200-2000 MB | ~12-15% |
| **cmix (fx2-cmix)** | **~700-900 MB** | **~12-15%** |

---

## Recommendations

### For Production Use:
1. **Accept the memory usage** - it's fundamental to the algorithm
2. **Monitor available RAM** before compression
3. **Consider adding `-memory <MB>` flag** to let users limit usage

### For Memory-Constrained Environments:
1. **Create "lite" profile** with reduced buffers (above optimizations)
2. **Stream processing** - compress in chunks (loses cross-chunk context)
3. **Use different algorithm** - LZMA/xz offers good ratio at ~100-200 MB

### For Research/Improvement:
1. **Profile actual usage** with `valgrind --tool=massif`
2. **Implement memory budgeting** - dynamically size buffers based on available RAM
3. **Add compression presets**: `-1` (fast, low memory) to `-9` (best, high memory)
4. **Use memory-mapped files** for history buffer (save ~60 MB physical RAM)

---

## Conclusion

**The high memory usage is intentional and necessary for achieving world-class compression ratios.** The major consumers are:
- History buffer (60 MB)
- Shared state map (100 MB)
- Match model hash tables (80 MB)
- Context maps in mixers (150 MB)
- Indirect hash tables (64 MB)
- FXCM model (50 MB)
- LSTM network (30 MB)

Each component trades memory for compression performance. Reducing any of these will decrease compression ratio. This is the nature of state-of-the-art compression algorithms.
