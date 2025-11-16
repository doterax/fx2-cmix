# UTF-8 Word Remapping Preprocessor

A lossless preprocessor that reduces file size by remapping frequent multi-byte UTF-8 words to unused single-byte ASCII slots.

## Overview

This preprocessor analyzes the input file to:
1. Identify unused ASCII byte slots (0-127)
2. Extract frequent multi-byte UTF-8 words using a trie-based collector
3. Replace the most impactful words with unused ASCII bytes
4. Store a compact mapping table in the output header

The approach is most effective on UTF-8 text with:
- Repeated multi-byte words (Chinese, Japanese, Korean, Cyrillic, etc.)
- Available unused ASCII slots in the byte range 0-127
- Sufficient word repetition to offset mapping table overhead

## File Format

### Compressed File Structure
```
[1 byte]  Slot count (N)
[N slots] Mapping table: for each slot:
            [1 byte] Slot byte value (unused ASCII)
            [1 byte] Word length in bytes
            [L bytes] Original UTF-8 word bytes
[remaining] Transformed data with words replaced by slot bytes
```

If slot count is 0, no transformation is applied (only 1 byte overhead).

## Building

```bash
make test-utf8-preprocess
```

This generates `test-utf8-preprocess.exe`.

## Usage

### Compress
```bash
test-utf8-preprocess c <input> <output>
```

Example:
```bash
test-utf8-preprocess c document.txt document.compressed
```

### Decompress
```bash
test-utf8-preprocess d <input> <output>
```

Example:
```bash
test-utf8-preprocess d document.compressed document_restored.txt
```

## Performance

Test results on UTF-8 benchmark files:

| File | Original Size | Compressed Size | Savings | Ratio |
|------|--------------|----------------|---------|-------|
| input | 51,052 bytes | 36,624 bytes | 14,428 bytes (28.26%) | 71.74% |
| input2 | 941,724 bytes | 844,922 bytes | 96,802 bytes (10.28%) | 89.72% |
| **enwik8** | **100,000,000 bytes** | **89,647,239 bytes** | **10,352,761 bytes (10.35%)** | **89.65%** |
| **enwik9** | **1,000,000,000 bytes** | **886,612,662 bytes** | **113,387,338 bytes (11.34%)** | **88.66%** |

All tests verified with SHA256 hash comparison (lossless round-trip).

Compression effectiveness varies based on:
- **Word frequency distribution**: High-frequency words yield better savings
- **Available unused ASCII**: More slots allow more word replacements
- **Word length**: Longer multi-byte words have higher impact scores
- **File size**: Larger files amortize mapping table overhead better

## Algorithm Details

### Compression
1. **ASCII Slot Detection**: Scan input for all used bytes < 128; remaining slots available for remapping.
2. **Word Collection**: Build a byte-level trie tracking word occurrences and lengths.
3. **Scoring**: Rank words by score = (word_length - 1) × occurrence_count.
4. **Mapping**: Assign top N words to N available slots.
5. **Transformation**: Replace each word occurrence with its assigned slot byte.

### Decompression
1. Read slot count and mapping table.
2. For each byte: if it's a slot, emit the original word; otherwise, emit the byte.

## Implementation Files

- `src/preprocess_utf8/utf8_preprocessor.h` - Public API
- `src/preprocess_utf8/utf8_preprocessor.cpp` - Core implementation
- `src/test_utf8_preprocess.cpp` - Command-line test tool

## Notes

- The preprocessor is **lossless**: decompression perfectly restores the original.
- Mapping table overhead: 1 + N × (2 + avg_word_length) bytes, where N ≤ unused ASCII slots.
- Word boundaries detected via Unicode separators (spaces, punctuation, control characters).
- Invalid UTF-8 sequences treated as separator boundaries.
- Maximum 255 words can be remapped (limited by slot count byte).
