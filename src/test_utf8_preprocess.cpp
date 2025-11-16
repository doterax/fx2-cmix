#include "preprocess_utf8/utf8_preprocessor.h"
#include <cstdio>
#include <cstring>

void PrintUsage(const char* program) {
    std::fprintf(stderr, "Usage:\n");
    std::fprintf(stderr, "  Compress:   %s c <input> <output>\n", program);
    std::fprintf(stderr, "  Decompress: %s d <input> <output>\n", program);
    std::fprintf(stderr, "\nExample:\n");
    std::fprintf(stderr, "  %s c input.txt compressed.dat\n", program);
    std::fprintf(stderr, "  %s d compressed.dat restored.txt\n", program);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        PrintUsage(argv[0]);
        return 1;
    }
    
    const char* mode = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];
    
    bool compress_mode = false;
    
    if (std::strcmp(mode, "c") == 0) {
        compress_mode = true;
    } else if (std::strcmp(mode, "d") == 0) {
        compress_mode = false;
    } else {
        std::fprintf(stderr, "Error: Invalid mode '%s'. Use 'c' for compress or 'd' for decompress.\n", mode);
        PrintUsage(argv[0]);
        return 1;
    }
    
    FILE* input = std::fopen(input_path, "rb");
    if (!input) {
        std::fprintf(stderr, "Error: Cannot open input file: %s\n", input_path);
        return 1;
    }
    
    FILE* output = std::fopen(output_path, "wb");
    if (!output) {
        std::fprintf(stderr, "Error: Cannot create output file: %s\n", output_path);
        std::fclose(input);
        return 1;
    }
    
    bool success = false;
    
    if (compress_mode) {
        std::fprintf(stderr, "Compressing: %s -> %s\n", input_path, output_path);
        success = utf8_preprocess::Compress(input, output);
        if (success) {
            std::fprintf(stderr, "Compression completed successfully.\n");
        } else {
            std::fprintf(stderr, "Error: Compression failed.\n");
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
