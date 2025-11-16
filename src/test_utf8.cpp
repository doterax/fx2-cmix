#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#undef min
#undef max
#endif

// Decode a UTF-8 sequence starting at the given byte pointer
// Returns the decoded codepoint and advances the pointer
// Returns 0 on invalid sequence
uint32_t DecodeUTF8(const unsigned char* data, size_t remaining, size_t* bytes_consumed) {
    if (remaining == 0) {
        *bytes_consumed = 0;
        return 0;
    }
    
    unsigned char first = data[0];
    
    // 1-byte sequence (ASCII): 0xxxxxxx
    if ((first & 0x80) == 0) {
        *bytes_consumed = 1;
        return first;
    }
    
    // 2-byte sequence: 110xxxxx 10xxxxxx
    if ((first & 0xE0) == 0xC0) {
        if (remaining < 2) {
            *bytes_consumed = 0;
            return 0;
        }
        if ((data[1] & 0xC0) != 0x80) {
            *bytes_consumed = 0;
            return 0;
        }
        *bytes_consumed = 2;
        return ((first & 0x1F) << 6) | (data[1] & 0x3F);
    }
    
    // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
    if ((first & 0xF0) == 0xE0) {
        if (remaining < 3) {
            *bytes_consumed = 0;
            return 0;
        }
        if ((data[1] & 0xC0) != 0x80 || (data[2] & 0xC0) != 0x80) {
            *bytes_consumed = 0;
            return 0;
        }
        *bytes_consumed = 3;
        return ((first & 0x0F) << 12) | ((data[1] & 0x3F) << 6) | (data[2] & 0x3F);
    }
    
    // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if ((first & 0xF8) == 0xF0) {
        if (remaining < 4) {
            *bytes_consumed = 0;
            return 0;
        }
        if ((data[1] & 0xC0) != 0x80 || (data[2] & 0xC0) != 0x80 || (data[3] & 0xC0) != 0x80) {
            *bytes_consumed = 0;
            return 0;
        }
        *bytes_consumed = 4;
        return ((first & 0x07) << 18) | ((data[1] & 0x3F) << 12) | 
               ((data[2] & 0x3F) << 6) | (data[3] & 0x3F);
    }
    
    // Invalid UTF-8 start byte
    *bytes_consumed = 0;
    return 0;
}

// Get the byte length of a UTF-8 codepoint
size_t GetUTF8ByteLength(uint32_t codepoint) {
    if (codepoint < 0x80) return 1;
    if (codepoint < 0x800) return 2;
    if (codepoint < 0x10000) return 3;
    return 4;
}

struct CharStats {
    uint32_t codepoint;
    size_t occurrences;
    size_t byte_length;
    size_t score; // occurrences * byte_length
};

// Windows console helpers for proper Unicode output
#ifdef _WIN32
static bool g_is_console = false;
static HANDLE g_stdout_handle = INVALID_HANDLE_VALUE;

void InitConsoleUTF8() {
    g_stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (g_stdout_handle == INVALID_HANDLE_VALUE) return;
    DWORD fileType = GetFileType(g_stdout_handle);
    if (fileType == FILE_TYPE_CHAR) {
        g_is_console = true;
        // Set console to UTF-8 code page; WriteConsoleW will still use UTF-16
        SetConsoleOutputCP(CP_UTF8);
    }
}

// Print a single Unicode codepoint to console preserving alignment.
// width = padding width in monospace columns (approximate; treat each as width 1 or 2 for surrogate pair)
void PrintCodepointCell(uint32_t cp, int width) {
    if (!g_is_console) {
        // Fallback: encode UTF-8 and print bytes; terminal may mojibake but hex column helps.
        char utf8[5] = {0};
        int len = 0;
        if (cp < 0x80) { utf8[0] = (char)cp; len = 1; }
        else if (cp < 0x800) { utf8[0] = (char)(0xC0 | (cp >> 6)); utf8[1] = (char)(0x80 | (cp & 0x3F)); len = 2; }
        else if (cp < 0x10000) { utf8[0] = (char)(0xE0 | (cp >> 12)); utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); utf8[2] = (char)(0x80 | (cp & 0x3F)); len = 3; }
        else { utf8[0] = (char)(0xF0 | (cp >> 18)); utf8[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); utf8[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); utf8[3] = (char)(0x80 | (cp & 0x3F)); len = 4; }
        printf("%-8s", utf8); // pad to width 8 for consistency
        return;
    }
    wchar_t wbuf[3];
    int wlen = 0;
    if (cp <= 0xFFFF) {
        wbuf[wlen++] = (wchar_t)cp;
    } else { // surrogate pair
        cp -= 0x10000;
        wbuf[wlen++] = (wchar_t)(0xD800 + (cp >> 10));
        wbuf[wlen++] = (wchar_t)(0xDC00 + (cp & 0x3FF));
    }
    DWORD written = 0;
    WriteConsoleW(g_stdout_handle, wbuf, wlen, &written, NULL);
    // pad spaces to maintain column width (assume each codepoint occupies width 1 visually; emojis may occupy 2)
    int pad = width - wlen; if (pad < 0) pad = 0;
    for (int i = 0; i < pad; ++i) putchar(' ');
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    InitConsoleUTF8();
#endif
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    FILE* input = fopen(argv[1], "rb");
    if (!input) {
        fprintf(stderr, "Error: Cannot open file: %s\n", argv[1]);
        return 1;
    }
    
    // Read entire file into memory for easier processing
    fseek(input, 0, SEEK_END);
    size_t file_size = ftell(input);
    fseek(input, 0, SEEK_SET);
    
    std::vector<unsigned char> buffer(file_size);
    size_t bytes_read = fread(buffer.data(), 1, file_size, input);
    fclose(input);
    
    if (bytes_read != file_size) {
        fprintf(stderr, "Error: Failed to read entire file\n");
        return 1;
    }
    
    // Statistics map: codepoint -> occurrence count
    std::map<uint32_t, size_t> stats;
    
    // Process file byte by byte, decoding UTF-8 sequences
    size_t pos = 0;
    size_t total_chars = 0;
    size_t invalid_sequences = 0;
    
    while (pos < file_size) {
        size_t bytes_consumed = 0;
        uint32_t codepoint = DecodeUTF8(buffer.data() + pos, file_size - pos, &bytes_consumed);
        
        if (bytes_consumed == 0) {
            // Invalid UTF-8 sequence - skip this byte
            invalid_sequences++;
            pos++;
            continue;
        }
        
        stats[codepoint]++;
        total_chars++;
        pos += bytes_consumed;
    }
    
    printf("UTF-8 Analysis Results\n");
    printf("======================\n");
    printf("Total file size: %zu bytes\n", file_size);
    printf("Total characters decoded: %zu\n", total_chars);
    printf("Invalid UTF-8 sequences: %zu\n", invalid_sequences);
    printf("Unique codepoints: %zu\n\n", stats.size());
    
    // Filter out ASCII characters (single-byte UTF-8)
    // and create CharStats entries for multi-byte characters
    std::vector<CharStats> multi_byte_chars;
    
    for (const auto& entry : stats) {
        uint32_t codepoint = entry.first;
        size_t occurrences = entry.second;
        size_t byte_length = GetUTF8ByteLength(codepoint);
        
        // Only include multi-byte UTF-8 characters (non-ASCII)
        if (byte_length > 1) {
            CharStats cs;
            cs.codepoint = codepoint;
            cs.occurrences = occurrences;
            cs.byte_length = byte_length;
            cs.score = occurrences * (byte_length - 1); // impact score
            multi_byte_chars.push_back(cs);
        }
    }
    
    // Sort by score (occurrences * (byte_length - 1)) in descending order
    std::sort(multi_byte_chars.begin(), multi_byte_chars.end(),
              [](const CharStats& a, const CharStats& b) {
                  return a.score > b.score;
              });
    
    // Print top 32 results
    printf("Top 32 Multi-byte UTF-8 Characters (by impact: occurrences * byte_length)\n");
    printf("===================================================================================================\n");
    printf("Rank  Codepoint  Char      Hex Bytes         Bytes  Occurrences  Impact\n");
    printf("----  ---------  --------  ----------------  -----  -----------  ------\n");
    
    size_t limit = (std::min)(size_t(32), multi_byte_chars.size());
    for (size_t i = 0; i < limit; i++) {
        const CharStats& cs = multi_byte_chars[i];
        
        // Encode the codepoint back to UTF-8 for display
        char utf8_char[5] = {0};
        char hex_bytes[17] = {0};
        
        if (cs.byte_length == 2) {
            utf8_char[0] = 0xC0 | ((cs.codepoint >> 6) & 0x1F);
            utf8_char[1] = 0x80 | (cs.codepoint & 0x3F);
            snprintf(hex_bytes, sizeof(hex_bytes), "%02X %02X", 
                     (unsigned char)utf8_char[0], (unsigned char)utf8_char[1]);
        } else if (cs.byte_length == 3) {
            utf8_char[0] = 0xE0 | ((cs.codepoint >> 12) & 0x0F);
            utf8_char[1] = 0x80 | ((cs.codepoint >> 6) & 0x3F);
            utf8_char[2] = 0x80 | (cs.codepoint & 0x3F);
            snprintf(hex_bytes, sizeof(hex_bytes), "%02X %02X %02X", 
                     (unsigned char)utf8_char[0], (unsigned char)utf8_char[1],
                     (unsigned char)utf8_char[2]);
        } else if (cs.byte_length == 4) {
            utf8_char[0] = 0xF0 | ((cs.codepoint >> 18) & 0x07);
            utf8_char[1] = 0x80 | ((cs.codepoint >> 12) & 0x3F);
            utf8_char[2] = 0x80 | ((cs.codepoint >> 6) & 0x3F);
            utf8_char[3] = 0x80 | (cs.codepoint & 0x3F);
            snprintf(hex_bytes, sizeof(hex_bytes), "%02X %02X %02X %02X", 
                     (unsigned char)utf8_char[0], (unsigned char)utf8_char[1],
                     (unsigned char)utf8_char[2], (unsigned char)utf8_char[3]);
        }
        
        printf("%4zu  U+%06X  ", i + 1, cs.codepoint);
    #ifdef _WIN32
        PrintCodepointCell(cs.codepoint, 8);
    #else
        printf("%-8s", utf8_char);
    #endif
        printf("  %-16s  %5zu  %11zu  %6zu\n", hex_bytes, cs.byte_length, cs.occurrences, cs.score);
    }
    
    if (multi_byte_chars.empty()) {
        printf("(No multi-byte UTF-8 characters found)\n");
    }

    // Report unused ASCII slots (bytes < 128) that did not appear in the input
    bool used_ascii[128] = {false};
    for (const auto &kv : stats) {
        uint32_t cp = kv.first;
        if (cp < 128) used_ascii[cp] = true;
    }

    std::vector<int> unused_ascii;
    for (int b = 0; b < 128; ++b) {
        if (!used_ascii[b]) unused_ascii.push_back(b);
    }

    printf("\nUnused ASCII byte slots (0..127): %zu\n", unused_ascii.size());
    if (!unused_ascii.empty()) {
        printf("Index  Dec   Hex  Char\n");
        printf("-----  ----  ---  ----\n");
        // Print in rows of 8 for readability
        for (size_t i = 0; i < unused_ascii.size(); ++i) {
            int b = unused_ascii[i];
            char preview;
            if (b >= 32 && b <= 126) preview = (char)b; else preview = '.';
            printf("%5d  %4d  %02X   %c\n", b, b, b & 0xFF, preview);
        }
    }

    // Compute potential impact if we remap the top K multi-byte chars to unused ASCII slots.
    size_t remap_capacity = (std::min)(unused_ascii.size(), multi_byte_chars.size());
    unsigned long long potential_saved = 0ULL;
    for (size_t i = 0; i < remap_capacity; ++i) {
        const CharStats &cs = multi_byte_chars[i];
        // Saving per occurrence: original UTF-8 length minus 1 (new single byte)
        potential_saved += (unsigned long long)cs.occurrences * (unsigned long long)(cs.byte_length - 1);
    }
    unsigned long long transformed_size = (potential_saved > file_size) ? 0 : (file_size - potential_saved);
    double pct_gain = file_size ? (100.0 * potential_saved / (double)file_size) : 0.0;

    printf("\nPotential remap analysis:\n");
    printf("  Remap capacity (unused ASCII slots): %zu\n", unused_ascii.size());
    printf("  Multi-byte candidates available     : %zu\n", multi_byte_chars.size());
    printf("  Characters considered for remap     : %zu\n", remap_capacity);
    printf("  Total bytes saved (ideal)          : %llu\n", potential_saved);
    printf("  Original size                      : %llu\n", (unsigned long long)file_size);
    printf("  Transformed size (ideal)           : %llu\n", transformed_size);
    printf("  Compression gain                   : %.2f%%\n", pct_gain);
    if (remap_capacity == 0) {
        printf("  (No available mapping opportunities)\n");
    } else {
        printf("  NOTE: Does not include overhead for storing a remap table.\n");
    }
    
    return 0;
}
