#ifndef UTF8_PREPROCESSOR_H
#define UTF8_PREPROCESSOR_H

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

namespace utf8_preprocess {

// Compress input file by replacing frequent multi-byte words with unused ASCII slots
// Returns true on success
bool Compress(FILE* input, FILE* output);

// Decompress by restoring original words from single-byte replacements
// Returns true on success
bool Decompress(FILE* input, FILE* output);

} // namespace utf8_preprocess

#endif // UTF8_PREPROCESSOR_H
