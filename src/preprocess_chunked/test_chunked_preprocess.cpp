#include "chunked_preprocessor.h"
#include <cstdio>
#include <cstring>

void PrintUsage(const char* program_name) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s c <input> <output>  - Compress using chunked preprocessing\n", program_name);
    fprintf(stderr, "  %s d <input> <output>  - Decompress using chunked preprocessing\n", program_name);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        PrintUsage(argv[0]);
        return 1;
    }
    
    const char* mode = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];
    
    FILE* input = fopen(input_path, "rb");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return 1;
    }
    
    FILE* output = fopen(output_path, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", output_path);
        fclose(input);
        return 1;
    }
    
    bool success = false;
    
    if (strcmp(mode, "c") == 0) {
        printf("Compressing '%s' to '%s'...\n", input_path, output_path);
        success = chunked_preprocess::Compress(input, output);
    } else if (strcmp(mode, "d") == 0) {
        printf("Decompressing '%s' to '%s'...\n", input_path, output_path);
        success = chunked_preprocess::Decompress(input, output);
    } else {
        fprintf(stderr, "Error: Unknown mode '%s'\n", mode);
        PrintUsage(argv[0]);
        fclose(input);
        fclose(output);
        return 1;
    }
    
    fclose(input);
    fclose(output);
    
    if (success) {
        printf("Operation completed successfully.\n");
        return 0;
    } else {
        fprintf(stderr, "Error: Operation failed.\n");
        return 1;
    }
}
