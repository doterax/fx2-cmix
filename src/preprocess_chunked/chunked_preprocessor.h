#ifndef CHUNKED_PREPROCESSOR_H
#define CHUNKED_PREPROCESSOR_H

#include <cstdint>
#include <cstdio>

namespace chunked_preprocess {

// Compress input file using chunked preprocessing with region detection
// Returns true on success
bool Compress(FILE* input, FILE* output);

// Decompress by restoring original structure from chunks
// Returns true on success
bool Decompress(FILE* input, FILE* output);

} // namespace chunked_preprocess

#endif // CHUNKED_PREPROCESSOR_H
