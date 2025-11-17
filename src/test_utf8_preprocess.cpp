#include "preprocess_utf8/utf8_preprocessor.h"
#include <cstdio>
#include <cstring>

void PrintUsage(const char *program) {
  std::fprintf(stderr, "Usage:\n");
  std::fprintf(stderr, "  CompressWords:   %s w <input> <output>\n", program);
  std::fprintf(stderr, "  CompressUtf8:    %s c <input> <output>\n", program);
  std::fprintf(stderr, "  Decompress:      %s d <input> <output>\n", program);
  std::fprintf(stderr, "\nExample:\n");
  std::fprintf(stderr, "  %s w input.txt compressed_words.dat\n", program);
  std::fprintf(stderr, "  %s c input.txt compressed_utf8.dat\n", program);
  std::fprintf(stderr, "  %s d compressed.dat restored.txt\n", program);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  const char *mode          = argv[1];
  const char *input_path    = argv[2];
  const char *output_path   = argv[3];

  enum Mode { COMPRESS_WORDS, COMPRESS_UTF8, DECOMPRESS };
  Mode op_mode;

  if (std::strcmp(mode, "w") == 0) {
    op_mode = COMPRESS_WORDS;
  } else if (std::strcmp(mode, "c") == 0) {
    op_mode = COMPRESS_UTF8;
  } else if (std::strcmp(mode, "d") == 0) {
    op_mode = DECOMPRESS;
  } else {
    std::fprintf(stderr,
                 "Error: Invalid mode '%s'. Use 'w' for word compress, 'c' for UTF-8 compress, or 'd' for decompress.\n",
                 mode);
    PrintUsage(argv[0]);
    return 1;
  }

  FILE *input = std::fopen(input_path, "rb");
  if (!input) {
    std::fprintf(stderr, "Error: Cannot open input file: %s\n", input_path);
    return 1;
  }

  FILE *output = std::fopen(output_path, "wb");
  if (!output) {
    std::fprintf(stderr, "Error: Cannot create output file: %s\n", output_path);
    std::fclose(input);
    return 1;
  }

  bool success = false;

  if (op_mode == COMPRESS_WORDS) {
    std::fprintf(stderr, "Compressing (words): %s -> %s\n", input_path, output_path);
    success = utf8_preprocess::CompressWords(input, output);
    if (success) {
      std::fprintf(stderr, "Word compression completed successfully.\n");
    } else {
      std::fprintf(stderr, "Error: Word compression failed.\n");
    }
  } else if (op_mode == COMPRESS_UTF8) {
    std::fprintf(stderr, "Compressing (UTF-8 chars): %s -> %s\n", input_path, output_path);
    success = utf8_preprocess::CompressUtf8(input, output);
    if (success) {
      std::fprintf(stderr, "UTF-8 compression completed successfully.\n");
    } else {
      std::fprintf(stderr, "Error: UTF-8 compression failed.\n");
    }
  } else {
    std::fprintf(stderr, "Decompressing: %s -> %s\n", input_path, output_path);
    success = utf8_preprocess::Decompress(input, output);
    if (success) {
      std::fprintf(stderr, "Decompression completed successfully.\n");
    } else {
      std::fprintf(stderr, "Error: Decompression failed.\n");
    }
  }

  std::fclose(input);
  std::fclose(output);

  return success ? 0 : 1;
}
