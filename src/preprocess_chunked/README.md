# Chunked Preprocessor

## Overview

The chunked preprocessor is a structure-aware text preprocessing system designed to preserve semantic information while making text more compressible for downstream algorithms (PPMd, dictionary compression, etc.).

Unlike simple character remapping approaches that obfuscate text structure, the chunked preprocessor identifies semantic regions and applies reversible transformations that maintain the underlying patterns that compression algorithms rely on.

## Architecture

### Region Types (8 types, 3 bits)

1. **REGION_RAW (0)**: Raw data copied as-is, no transformation
2. **REGION_ASCII_SENTENCE (1)**: ASCII sentence ending with period
   - Stores lowercase version without trailing dot
   - Transformation indicates original case pattern
3. **REGION_BRACKETS (2)**: Wikipedia-style references `[[...]]`
   - Stores only inner content, outer brackets reconstructed
4. **REGION_WIKI_HEADER (3)**: Wiki section headers `==...==`
   - Stores only inner text, `==` markers reconstructed
5. **REGION_XML_TAG (4)**: Simple XML tags `<tag>...</tag>`
   - Stores tag name + separator + content
   - Transformations apply to content only
6. **REGION_CURLY_BRACKETS (5)**: Template references `{{...}}`
   - Stores only inner content, outer brackets reconstructed
7. **REGION_UTF8_COMPRESSED (6)**: UTF-8 character sequences with small codepoint range
   - When max_codepoint - min_codepoint ≤ 255
   - Stores min_codepoint (UTF-8) + byte offsets
8. **REGION_RESERVED (7)**: Reserved for future use

### Transformations (4 types, 2 bits)

1. **TRANSFORM_NONE (0)**: No transformation applied
2. **TRANSFORM_FIRST_UPPER (1)**: First ASCII character was uppercase
   - Stored lowercase, restored to uppercase on decompression
3. **TRANSFORM_ALL_UPPER (2)**: All ASCII characters were uppercase
   - Stored lowercase, restored to uppercase on decompression
4. **TRANSFORM_RESERVED (3)**: Reserved for future use

### Chunk Header Format

**16-bit header (default):**
- Byte 0:
  - Bits 0-3: Region type (4 bits)
  - Bits 4-5: Transformation (2 bits)
  - Bit 6: bigSize flag (0 = 8-bit size follows, 1 = 16-bit size follows)
  - Bit 7: contextDependent flag (reserved for future use)
- Byte 1: Size (8 bits) if bigSize=0
  
**24-bit header (for chunks > 255 bytes):**
- Byte 0: Same as above with bigSize=1
- Byte 1-2: Size (16 bits, big-endian)

## Implementation Details

### Detection Rules

#### ASCII Sentence
- Must start with uppercase letter (A-Z)
- Must end with period (.)
- Can contain: letters, spaces, commas, semicolons, colons
- Two patterns:
  1. Mixed case: starts uppercase, contains lowercase
  2. All uppercase: all letters are uppercase (with punctuation)

#### Brackets/Headers/Curly Brackets
- Simple pattern matching: `[[...]]`, `==...==`, `{{...}}`
- Extracts inner content only
- Applies transformation to inner content

#### XML Tags
- Pattern: `<lowercase_tag>...</lowercase_tag>`
- Tag must be lowercase letters only (no attributes)
- Stores: tag_name + 0x00 separator + content
- Transformation applies to content

#### UTF-8 Compressed
- Scans ahead collecting UTF-8 characters (codepoint >= 0x80)
- Stops when:
  - ASCII character encountered
  - max_codepoint - min_codepoint > 255
  - 256 characters collected
- Requires minimum 3 UTF-8 characters
- Encoding: min_codepoint (UTF-8 1-4 bytes) + deltas (1 byte each)

### Transformation Application

Transformations are only applied to detected regions, preserving case information in a compact form:

1. **First Uppercase**: Common for sentence starts
   - "Hello world" → type=sentence, transform=first_upper, data="hello world"
   
2. **All Uppercase**: Common for acronyms, headers
   - "TITLE TEXT" → type=wiki_header, transform=all_upper, data="title text"

3. **Fallback to RAW**: If transformation would increase size, falls back to raw region

## Usage

### Compilation
```bash
make test-chunked-preprocess
```

### Compression
```bash
./test-chunked-preprocess c input.txt output.bin
```

### Decompression
```bash
./test-chunked-preprocess d output.bin recovered.txt
```

### Verification
```bash
# PowerShell
$original = [System.IO.File]::ReadAllBytes("input.txt")
$decompressed = [System.IO.File]::ReadAllBytes("recovered.txt")
Compare-Object $original $decompressed
```

## Design Rationale

### Why Not Character Remapping?

Previous approaches (CompressWords, CompressUtf8, CompactUtf8) remapped frequent characters to unused ASCII slots. This provided ~10% size reduction but had critical flaws:

1. **Obfuscates Text Structure**: PPMd and dictionary compression rely on recognizing English words and patterns. Remapping destroys these patterns.

2. **Breaks Text Detection**: The preprocessor's text detector is optimized for English byte distributions. Remapped bytes confuse detection.

3. **Context Destruction**: Word boundaries and character frequencies become meaningless after remapping.

### Why Chunked Preprocessing?

The chunked approach:

1. **Preserves Structure**: English text remains recognizable to downstream algorithms
2. **Compact Metadata**: Case transformations store 2 bits instead of full case information
3. **Semantic Awareness**: Different text types (sentences, headers, references) handled appropriately
4. **Incremental Processing**: Chunks can be processed sequentially without loading entire file
5. **Extensible**: 8 region types and 4 transformations allow future enhancements

### Expected Benefits

While the preprocessing itself adds ~1-3% overhead from chunk headers, it makes text more compressible for downstream algorithms:

1. **Lowercase Normalization**: Reduces alphabet size from 52 (A-Z, a-z) to 26 letters
2. **Structural Markers Removed**: `[[`, `]]`, `==`, `{{`, `}}` removed from stream
3. **UTF-8 Range Compression**: Multi-byte UTF-8 reduced to 1 byte per character when codepoint range is small
4. **Sentence Boundaries**: Trailing dots removed, reducing punctuation entropy

The real compression happens when this preprocessed data is fed to PPMd, which can now:
- Build better context models on normalized lowercase text
- Recognize repeated patterns without case variations
- Model structural elements (brackets, headers) separately

## Testing

### Small Test (196 bytes)
```
Original: 196 bytes
Preprocessed: 199 bytes (101.53%)
Roundtrip: ✓ Identical
```

### Larger Test (1070 bytes)
```
Original: 1070 bytes
Preprocessed: 1081 bytes (101.03%)
Roundtrip: ✓ Identical
```

The small overhead is expected - the value comes from improved downstream compression, not from the preprocessing itself.

## Future Enhancements

1. **Additional Region Types**: Could detect more patterns (lists, tables, code blocks)
2. **Context-Dependent Flags**: Use reserved flag for context-aware processing
3. **Dictionary Integration**: Pass detected words to dictionary compressor
4. **Multi-Pass**: First pass detects regions, second pass optimizes encoding
5. **Adaptive Thresholds**: Adjust UTF-8 compression threshold based on content

## Files

- `src/preprocess_chunked/chunked_preprocessor.h` - Header with function declarations
- `src/preprocess_chunked/chunked_preprocessor.cpp` - Implementation
- `src/preprocess_chunked/test_chunked_preprocess.cpp` - Test program
- `makefile` - Build targets: `make test-chunked-preprocess`
