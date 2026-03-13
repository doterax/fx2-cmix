# Dictionary Compress — Analysis & Results

## Results Summary

### enwik9 (1 GB) — cmix store verified

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|------------------|
| Raw (no preprocessing) | 1,000,000,000 | 5.1565 | 644,561,309 |
| cmix store (english.dic) | 647,798,597 | 6.2242 | 504,003,449 |
| cmix store (enwik9_optimal.dic) | 631,327,931 | 6.1409 | 484,614,664 |
| Our tool (english.dic) | 647,798,586 | 6.2242 | 504,003,422 |
| **Our tool (enwik9_optimal.dic)** | **631,327,920** | **6.1409** | **484,614,638** |

**New dictionary saves ~19.4 MB** estimated compressed size vs english.dic on enwik9.
Our tool matches cmix store exactly (~11 bytes difference = segment header).

### enwik8 (100 MB)

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|-----------------|
| Raw (no preprocessing) | 100,000,000 | 5.0801 | 63,501,754 |
| english.dic (original cmix) | 61,593,203 | 6.2386 | 48,032,000 |
| **words_enwik9_optimal.dic (new)** | **61,613,505** | **6.1842** | **47,628,867** |

**New dictionary saves ~403 KB** estimated compressed size vs original on enwik8.

### enwik7 (10 MB)

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|-----------------|
| Raw (no preprocessing) | 10,000,000 | 5.0993 | 6,374,151 |
| english.dic (original cmix) | 6,251,846 | 6.2182 | 4,859,435 |
| **words_enwik9_optimal.dic (new)** | **6,265,661** | **6.1653** | **4,828,731** |

**New dictionary saves ~31 KB** estimated compressed size vs original on enwik7.

### Verification: our tool matches cmix exactly

**enwik9** (1 GB):

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|------------------|
| cmix store (english.dic) | 647,798,597 | 6.2242 | 504,003,449 |
| Our tool (english.dic) | 647,798,586 | 6.2242 | 504,003,422 |
| cmix store (enwik9_optimal.dic) | 631,327,931 | 6.1409 | 484,614,664 |
| Our tool (enwik9_optimal.dic) | 631,327,920 | 6.1409 | 484,614,638 |

**enwik7** (10 MB):

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|------------------|
| cmix store (english.dic) | 6,251,857 | 6.2183 | 4,859,455 |
| Our tool (english.dic) | 6,251,846 | 6.2182 | 4,859,435 |
| **Our tool (enwik9_optimal.dic)** | **6,265,661** | **6.1653** | **4,828,731** |

The 11-byte size difference is the segment header (5 bytes) + flag byte (1 byte) that cmix writes.
Verified on both enwik7 and enwik9 — our tool produces identical dictionary encoding to cmix.

### IMPORTANT: The "3.67 entropy" figure was a bug

Earlier analysis showed cmix preprocessing at 3.67 bits/byte, but this was caused by a
**bug in cmix's dictionary parser** (undefined behavior). The file `words_enwik8.dic` has
514,238 word entries — far exceeding the 44,880 max. cmix's Dictionary constructor has
no bounds check and uses an uninitialized `bytes` variable for entries beyond the limit:

```cpp
// From dictionary.cpp lines 50-65:
unsigned int bytes;                    // ← NEVER INITIALIZED
if (line_count < kBoundary1) {
    bytes = 0x80 + line_count;
} else if (line_count < kBoundary2) {
    bytes = ...;
} else if (line_count < kBoundary3) {  // 44880
    bytes = ...;
}
// When line_count >= 44880: NONE of the above assign bytes → UB!
byte_map_[line] = bytes;               // ← STORES GARBAGE
++line_count;                          // ← NO BREAK, keeps going
```

This corrupts the code map for words that appear again after index 44880, creating
unpredictable (but coincidentally low-entropy) output. The dictionaries affected:

| Dictionary | Word entries | Status |
|-----------|-------------|--------|
| english.dic | 44,515 | OK |
| words_enwik9_optimal.dic | 44,880 | OK |
| words_enwik8.dic | 514,238 | **OVERFLOW (UB!)** |
| words_enwik8_opt.dic | 382,765 | **OVERFLOW (UB!)** |
| words_enwik9_not_pruned.dic | 1,524,017 | **OVERFLOW (UB!)** |
| words_enwik9.dic | 63,768 | **OVERFLOW (UB!)** |

When `words_enwik8.dic` is properly parsed (first 44,880 entries saved to clean format),
cmix itself produces 9,063,740 bytes at **6.2751 entropy** — NOT 3.67.

### Old Proposed Approach (base62 codes) — for reference

| Approach | Encoded Size | Entropy (bits/byte) | Est. Compressed |
|----------|-------------|---------------------|-----------------|
| Raw (no preprocessing) | 10,000,000 | 5.0993 | 6,374,151 |
| Old proposed (base62 markers) | 7,234,350 | 5.4669 | 4,943,836 |

The old base62 approach was **fundamentally worse** — higher entropy than raw input.

---

## New Dictionary Details

- **File**: `dictionary/words_enwik9_optimal.dic`
- **Entries**: 44,880
- **Built from**: `prof_input/enwik9` (1 GB Wikipedia dump)
- **Tier 1 (1-byte codes)**: 80 entries (most valuable words)
- **Tier 2 (2-byte codes)**: 3,840 entries
- **Tier 3 (3-byte codes)**: 40,960 entries
- **Estimated byte savings on training corpus**: 368,672,080 bytes (36.87%)

### Top 20 words (1-byte code slots)

| Slot | Word | Code Size |
|------|------|-----------|
| 0 | the | 1 byte |
| 1 | and | 1 byte |
| 2 | quot | 1 byte |
| 3 | contributor | 1 byte |
| 4 | of | 1 byte |
| 5 | timestamp | 1 byte |
| 6 | revision | 1 byte |
| 7 | category | 1 byte |
| 8 | username | 1 byte |
| 9 | in | 1 byte |
| 10 | population | 1 byte |
| 11 | comment | 1 byte |
| 12 | with | 1 byte |
| 13 | title | 1 byte |
| 14 | to | 1 byte |
| 15 | census | 1 byte |
| 16 | from | 1 byte |
| 17 | for | 1 byte |
| 18 | that | 1 byte |
| 19 | amp | 1 byte |

---

## Encoding Scheme

Uses the same high-byte encoding as cmix's original preprocessor (`src/preprocess/dictionary.cpp`):

### Code tiers

| Tier | Index Range | Byte Cost | First Byte | Second Byte | Third Byte |
|------|-------------|-----------|------------|-------------|------------|
| 1 | 0–79 | 1 byte | 0x80 + idx | — | — |
| 2 | 80–3,919 | 2 bytes | 0xD0 + hi | 0x80 + lo | — |
| 3 | 3,920–44,879 | 3 bytes | 0xF0 + hi | 0xD0 + mid | 0x80 + lo |

### Special markers

| Byte | Meaning |
|------|---------|
| 0x40 | Next word is Capitalized |
| 0x07 | Following words are UPPERCASE |
| 0x06 | End of UPPERCASE run |
| 0x0C | Escape: next byte is literal (not a marker/code) |

### Disambiguation

- First byte 0x80–0xCF → tier 1 (1-byte code, done)
- First byte 0xD0–0xFF → read second byte:
  - Second byte 0x80–0xCF → tier 2 (2-byte code, done)
  - Second byte 0xD0+ → read third byte → tier 3 (3-byte code)

---

## Why The Old Base62 Approach Failed

The previous `dictionary_compress.cpp` used printable base62 codes (`a-zA-Z0-9`) with marker prefixes (`~`, `^`, `*`, `\`). Five fundamental problems:

1. **Marker overhead**: +1 byte per word occurrence for the marker prefix, eating into savings
2. **Namespace collision**: base62 codes use `a-zA-Z0-9` which overlaps with text content, requiring markers to distinguish codes from words
3. **Entropy destruction**: Mixing codes with content creates a flat, noisy byte distribution (5.47 bits/byte — *worse* than raw input at 5.10)
4. **Short word penalty**: 2-letter words (of, in, to, on) get zero savings with a 1-char marker + 1-char code; 1-letter words actually expand
5. **No pretraining or byte remapping**: Missing the statistical optimizations cmix applies

### Why high-byte codes work

High-byte codes (0x80+) occupy a completely separate byte range from ASCII text (0x00–0x7F). This creates a **bimodal byte distribution** that arithmetic coders exploit efficiently — the model can predict whether the next byte is text or a code with high confidence.

---

## How The Optimal Dictionary Is Built

The `build-dict` command:

1. **Scan** the corpus and count every word's frequency (case-insensitive, splitting on uppercase→lowercase transitions)
2. **Compute savings** per word per tier: `savings = count × (word_length - code_byte_length)`
3. **Greedily fill** tier 1 (80 slots, 1-byte codes) with the highest-savings words
4. **Greedily fill** tier 2 (3,840 slots, 2-byte codes) with the next-best words
5. **Greedily fill** tier 3 (40,960 slots, 3-byte codes) with remaining beneficial words
6. **Prune** any word with savings ≤ 0 (would make output larger)

This ensures the 80 most valuable 1-byte slots go to words where `count × (len - 1)` is maximized — not just the most frequent words, but the ones that save the most total bytes.

---

## Usage

```bash
# Build optimized dictionary from corpus
dictionary_compress.exe build-dict --input prof_input/enwik9 --dict dictionary/words_enwik9_optimal.dic --verbose

# Preprocess (encode) a file
dictionary_compress.exe compress --input prof_input/enwik8 --dict dictionary/words_enwik9_optimal.dic --output encoded.bin

# Reverse preprocessing (decode)
dictionary_compress.exe decompress --input encoded.bin --dict dictionary/words_enwik9_optimal.dic --output decoded.txt

# Analyze without writing output
dictionary_compress.exe analyze --input prof_input/enwik8 --dict dictionary/words_enwik9_optimal.dic

# Verify roundtrip correctness
dictionary_compress.exe verify --input prof_input/enwik8 --dict dictionary/words_enwik9_optimal.dic
```

---

## Byte Distribution (enwik8, new dictionary)

```
Byte class distribution:
  Control (no ws):     892,957 ( 1.45%)
  Whitespace:       14,652,205 (23.78%)
  Printable ASCII:  22,908,303 (37.18%)
  High (>=0x80):    23,160,040 (37.59%)
```

Top bytes in encoded output:
```
  byte  32 (' '):   13,519,824 (21.94%)
  byte  64 ('@'):    3,138,840 ( 5.09%)   ← capitalized marker
  byte  93 (']'):    1,945,253 ( 3.16%)
  byte  91 ('['):    1,945,220 ( 3.16%)
  byte  10 (0x0A):   1,128,023 ( 1.83%)
  byte 128 (0x80):     956,305 ( 1.55%)   ← tier 1 code for "the"
  byte  39 ('''):      946,508 ( 1.54%)
  byte  97 ('a'):      881,158 ( 1.43%)
  byte  46 ('.'):      794,548 ( 1.29%)
  byte  44 (','):      787,826 ( 1.28%)
```

---

## Word Encoding Statistics (enwik8)

| Metric | english.dic | words_enwik9_optimal.dic |
|--------|-------------|--------------------------|
| Total words | 14,521,127 | 14,521,118 |
| Words encoded | 13,035,292 (89.8%) | 12,365,407 (85.2%) |
| Bytes saved | 38,406,797 | 38,386,495 |
| Output entropy | 6.2386 | **6.1842** |
| Est. compressed | 48,032,000 | **47,628,867** |

The new dictionary encodes fewer words (85.2% vs 89.8%) but achieves lower entropy because it assigns shorter codes to the most impactful words. Quality of code assignment matters more than quantity of words encoded.

---

## Additional Features

- **Substring matching**: Words longer than 7 characters that aren't in the dictionary are matched by prefix or suffix against dictionary entries, reducing the unknown-word penalty
- **Case handling**: Capitalized and UPPERCASE words are encoded as lowercase + case markers, avoiding duplicate dictionary entries
- **Escape mechanism**: Literal bytes that collide with markers or high-byte codes are escaped with 0x0C prefix
- **cmix-compatible dictionary format**: One lowercase word per line, position = code index — drop-in replacement for `english.dic`
