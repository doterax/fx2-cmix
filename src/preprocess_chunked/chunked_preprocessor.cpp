#include "chunked_preprocessor.h"
#include <vector>
#include <cstring>
#include <algorithm>

namespace chunked_preprocess {

// Region types (3 bits = 0-7)
enum RegionType : uint8_t {
    REGION_RAW = 0,
    REGION_ASCII_SENTENCE = 1,
    REGION_BRACKETS = 2,           // [[...]]
    REGION_WIKI_HEADER = 3,        // ==...==
    REGION_XML_TAG = 4,            // <tag>...</tag>
    REGION_CURLY_BRACKETS = 5,     // {{...}}
    REGION_UTF8_COMPRESSED = 6,
    REGION_RESERVED = 7
};

// Text transformations (2 bits)
enum Transformation : uint8_t {
    TRANSFORM_NONE = 0,
    TRANSFORM_FIRST_UPPER = 1,      // Make first ASCII char uppercase
    TRANSFORM_ALL_UPPER = 2,        // Make all ASCII chars uppercase
    TRANSFORM_RESERVED = 3
};

// Chunk header structure
struct ChunkHeader {
    RegionType type : 4;
    Transformation transform : 2;
    bool big_size : 1;
    bool context_dependent : 1;
    uint16_t size;  // 8 or 16 bits depending on big_size
};

// UTF-8 decoder
static uint32_t DecodeUTF8(const unsigned char* data, size_t remaining, size_t* bytes_consumed) {
    if (remaining == 0) { *bytes_consumed = 0; return 0; }
    unsigned char b0 = data[0];
    if ((b0 & 0x80) == 0) { *bytes_consumed = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (remaining < 2 || (data[1] & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 2;
        return ((b0 & 0x1F) << 6) | (data[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (remaining < 3 || (data[1] & 0xC0) != 0x80 || (data[2] & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 3;
        return ((b0 & 0x0F) << 12) | ((data[1] & 0x3F) << 6) | (data[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (remaining < 4 || (data[1] & 0xC0) != 0x80 || (data[2] & 0xC0) != 0x80 || (data[3] & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 4;
        return ((b0 & 0x07) << 18) | ((data[1] & 0x3F) << 12) | ((data[2] & 0x3F) << 6) | (data[3] & 0x3F);
    }
    *bytes_consumed = 0;
    return 0;
}

static size_t EncodeUTF8(uint32_t cp, unsigned char* out) {
    if (cp < 0x80) {
        out[0] = (unsigned char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = 0xC0 | ((cp >> 6) & 0x1F);
        out[1] = 0x80 | (cp & 0x3F);
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = 0xE0 | ((cp >> 12) & 0x0F);
        out[1] = 0x80 | ((cp >> 6) & 0x3F);
        out[2] = 0x80 | (cp & 0x3F);
        return 3;
    }
    out[0] = 0xF0 | ((cp >> 18) & 0x07);
    out[1] = 0x80 | ((cp >> 12) & 0x3F);
    out[2] = 0x80 | ((cp >> 6) & 0x3F);
    out[3] = 0x80 | (cp & 0x3F);
    return 4;
}

static bool IsUpperASCII(unsigned char c) {
    return c >= 'A' && c <= 'Z';
}

static bool IsLowerASCII(unsigned char c) {
    return c >= 'a' && c <= 'z';
}

static bool IsAlphaASCII(unsigned char c) {
    return IsUpperASCII(c) || IsLowerASCII(c);
}

static bool IsDigit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static bool IsAlphaNumASCII(unsigned char c) {
    return IsAlphaASCII(c) || IsDigit(c);
}

static unsigned char ToLower(unsigned char c) {
    if (IsUpperASCII(c)) return c + 32;
    return c;
}

static unsigned char ToUpper(unsigned char c) {
    if (IsLowerASCII(c)) return c - 32;
    return c;
}

static void WriteChunkHeader(FILE* output, const ChunkHeader& header) {
    uint8_t byte1 = (header.type & 0x0F) | ((header.transform & 0x03) << 4) |
                    (header.big_size ? 0x40 : 0) | (header.context_dependent ? 0x80 : 0);
    
    fputc(byte1, output);
    
    if (header.big_size) {
        fputc((header.size >> 8) & 0xFF, output);
        fputc(header.size & 0xFF, output);
    } else {
        fputc(header.size & 0xFF, output);
    }
}

static bool ReadChunkHeader(FILE* input, ChunkHeader& header) {
    int byte1 = fgetc(input);
    if (byte1 == EOF) return false;
    
    header.type = static_cast<RegionType>(byte1 & 0x0F);
    header.transform = static_cast<Transformation>((byte1 >> 4) & 0x03);
    header.big_size = (byte1 & 0x40) != 0;
    header.context_dependent = (byte1 & 0x80) != 0;
    if (header.big_size) {
        int hi = fgetc(input);
        int lo = fgetc(input);
        if (hi == EOF || lo == EOF) return false;
        header.size = (hi << 8) | lo;
    } else {
        int size_byte = fgetc(input);
        if (size_byte == EOF) return false;
        header.size = size_byte;
    }
    return true;
}

// Helper: find next potential structure boundary starting at 'from'
static size_t FindNextBoundary(const std::vector<unsigned char>& buffer, size_t from) {
    for (size_t i = from; i < buffer.size(); ++i) {
        unsigned char c = buffer[i];
        // Check for structure starts
        if (i + 1 < buffer.size()) {
            if ((c == '[' && buffer[i+1] == '[') || 
                (c == '=' && buffer[i+1] == '=') ||
                (c == '{' && buffer[i+1] == '{')) {
                return i;
            }
        }
        // Any '<' could start an XML tag
        if (c == '<') {
            return i;
        }
        if (IsUpperASCII(c)) {
            return i;
        }
        // Check for multi-byte UTF-8 start
        if ((c & 0x80) != 0) {
            return i;
        }
    }
    return buffer.size();
}

// Detect region type at current position
static RegionType DetectRegion(const std::vector<unsigned char>& buffer, size_t pos, size_t& region_end) {
    if (pos >= buffer.size()) return REGION_RAW;
    
    // Check for brackets [[...]]
    if (pos + 1 < buffer.size() && buffer[pos] == '[' && buffer[pos + 1] == '[') {
        for (size_t i = pos + 2; i + 1 < buffer.size(); ++i) {
            if (buffer[i] == ']' && buffer[i + 1] == ']') {
                region_end = i + 2;
                return REGION_BRACKETS;
            }
        }
    }
    
    // Check for wiki header ==...==
    if (pos + 1 < buffer.size() && buffer[pos] == '=' && buffer[pos + 1] == '=') {
        for (size_t i = pos + 2; i + 1 < buffer.size(); ++i) {
            if (buffer[i] == '=' && buffer[i + 1] == '=') {
                region_end = i + 2;
                return REGION_WIKI_HEADER;
            }
        }
    }
    
    // Check for curly brackets {{...}}
    if (pos + 1 < buffer.size() && buffer[pos] == '{' && buffer[pos + 1] == '{') {
        for (size_t i = pos + 2; i + 1 < buffer.size(); ++i) {
            if (buffer[i] == '}' && buffer[i + 1] == '}') {
                region_end = i + 2;
                return REGION_CURLY_BRACKETS;
            }
        }
    }
    
    // Check for XML tag <tag>text</tag> (no nested XML, no attributes)
    if (buffer[pos] == '<' && pos + 1 < buffer.size() && IsLowerASCII(buffer[pos + 1])) {
        size_t tag_start = pos + 1;
        size_t tag_end = tag_start;
        // tag name: lowercase letters only
        while (tag_end < buffer.size() && IsLowerASCII(buffer[tag_end])) tag_end++;
        
        // Reject attributes or invalid tag names
        if (tag_end < buffer.size() && buffer[tag_end] == '>') {
            size_t tag_len = tag_end - tag_start;
            if (tag_len > 0 && tag_len <= 255) {
                // content starts after '>'
                size_t content_start = tag_end + 1;
                // search for exact closing tag position
                // pattern: "</" + tag_name + ">"
                const size_t close_pat_extra = 3; // </ + >
                bool found = false;
                size_t close_start = 0;
                for (size_t i = content_start; i + tag_len + 2 < buffer.size(); ++i) {
                    if (buffer[i] == '<') {
                        // If this is the closing tag, accept; otherwise reject nested XML
                        if (i + 2 + tag_len < buffer.size() && buffer[i + 1] == '/' &&
                            std::equal(buffer.begin() + i + 2, buffer.begin() + i + 2 + tag_len, buffer.begin() + tag_start) &&
                            buffer[i + 2 + tag_len] == '>') {
                            close_start = i;
                            found = true;
                            break;
                        } else {
                            // Nested or unexpected '<' before closing tag => not a simple XML text node
                            found = false;
                            break;
                        }
                    }
                }
                if (found && close_start >= content_start) {
                    size_t content_len = close_start - content_start;
                    // Ensure content length fits in 16-bit chunk (since we use up to 2-byte size)
                    if (content_len <= 65535) {
                        region_end = close_start + tag_len + close_pat_extra; // position after closing tag 
                        return REGION_XML_TAG;
                    }
                }
            }
        }
    }
    
    // Check for ASCII sentence (starts with uppercase, ends with dot)
    if (IsUpperASCII(buffer[pos])) {
        size_t i = pos;
        bool has_lower = false;
        bool all_upper = true;
        
        while (i < buffer.size()) {
            unsigned char c = buffer[i];
            if (c == '.') {
                // Found sentence ending with dot
                region_end = i + 1;
                return REGION_ASCII_SENTENCE;
            }
            if (c == '\n' || c == '\r') {
                // Newline without dot - not a sentence, break
                break;
            }
            if (IsLowerASCII(c)) {
                has_lower = true;
                all_upper = false;
            } else if (!IsUpperASCII(c) && !IsAlphaNumASCII(c) && c != ' ' && c != ',' && c != ';' && c != ':' && c != '!' && c != '?') {
                // Non-sentence character
                break;
            }
            ++i;
        }
    }
    
    // Check for UTF-8 compressed region (multiple UTF-8 chars with small range)
    size_t utf8_count = 0;
    uint32_t min_cp = 0xFFFFFFFF;
    uint32_t max_cp = 0;
    size_t i = pos;
    
    while (i < buffer.size() && utf8_count < 256) {
        size_t consumed = 0;
        uint32_t cp = DecodeUTF8(buffer.data() + i, buffer.size() - i, &consumed);
        if (consumed == 0) break;
        if (cp < 0x80) break; // ASCII breaks UTF-8 region
        
        if (cp < min_cp) min_cp = cp;
        if (cp > max_cp) max_cp = cp;
        
        if (max_cp - min_cp > 255) break;
        
        utf8_count++;
        i += consumed;
    }
    
    if (utf8_count >= 3 && max_cp - min_cp <= 255) {
        region_end = i;
        return REGION_UTF8_COMPRESSED;
    }
    
    // RAW fallback: consume up to next recognizable boundary (max 255 bytes)
    // Search from pos+1 to avoid detecting current position as boundary
    size_t next_boundary = FindNextBoundary(buffer, pos + 1);
    if (next_boundary == pos + 1) {
        // Boundary found immediately - emit single byte
        region_end = pos + 1;
    } else {
        size_t raw_len = std::min(next_boundary - pos, (size_t)255);
        region_end = pos + raw_len;
    }
    return REGION_RAW;
}

// Apply transformation to text
static void ApplyTransform(std::vector<unsigned char>& text, Transformation transform) {
    if (transform == TRANSFORM_FIRST_UPPER && !text.empty()) {
        text[0] = ToLower(text[0]);
    } else if (transform == TRANSFORM_ALL_UPPER) {
        for (auto& c : text) {
            c = ToLower(c);
        }
    }
}

static void ReverseTransform(std::vector<unsigned char>& text, Transformation transform) {
    if (transform == TRANSFORM_FIRST_UPPER && !text.empty()) {
        text[0] = ToUpper(text[0]);
    } else if (transform == TRANSFORM_ALL_UPPER) {
        for (auto& c : text) {
            c = ToUpper(c);
        }
    }
}

bool Compress(FILE* input, FILE* output) {
    if (!input || !output) return false;
    
    // Read entire input
    fseek(input, 0, SEEK_END);
    size_t file_size = ftell(input);
    fseek(input, 0, SEEK_SET);
    
    std::vector<unsigned char> buffer(file_size);
    if (fread(buffer.data(), 1, file_size, input) != file_size) {
        return false;
    }
    
    size_t pos = 0;
    while (pos < file_size) {
        // Skip whitespace/newlines between chunks (encode as RAW if significant)
        size_t ws_start = pos;
        while (pos < file_size && (buffer[pos] == '\n' || buffer[pos] == '\r' || buffer[pos] == ' ')) {
            pos++;
        }
        
        // If we consumed whitespace, emit as RAW
        if (pos > ws_start) {
            ChunkHeader ws_header;
            ws_header.type = REGION_RAW;
            ws_header.transform = TRANSFORM_NONE;
            ws_header.context_dependent = false;
            size_t ws_len = pos - ws_start;
            ws_header.size = (uint16_t)ws_len;
            ws_header.big_size = ws_len > 255;
            
            WriteChunkHeader(output, ws_header);
            fwrite(buffer.data() + ws_start, 1, ws_len, output);
            
            if (pos >= file_size) break;
        }
        
        size_t region_end = pos + 1;
        RegionType type = DetectRegion(buffer, pos, region_end);
        
        std::vector<unsigned char> chunk_data;
        Transformation transform = TRANSFORM_NONE;
        
        if (type == REGION_ASCII_SENTENCE) {
            // Extract sentence, remove trailing dot
            chunk_data.assign(buffer.begin() + pos, buffer.begin() + region_end - 1);
            
            // Determine transformation
            if (!chunk_data.empty()) {
                bool first_is_upper = IsUpperASCII(chunk_data[0]);
                bool all_upper = true;
                for (auto c : chunk_data) {
                    if (IsLowerASCII(c)) {
                        all_upper = false;
                        break;
                    }
                }
                
                if (all_upper && chunk_data.size() > 1) {
                    transform = TRANSFORM_ALL_UPPER;
                } else if (first_is_upper) {
                    transform = TRANSFORM_FIRST_UPPER;
                }
            }
            
            ApplyTransform(chunk_data, transform);
            
        } else if (type == REGION_BRACKETS || type == REGION_CURLY_BRACKETS) {
            // Extract content between brackets (skip outer 2 chars on each side)
            chunk_data.assign(buffer.begin() + pos + 2, buffer.begin() + region_end - 2);
            
            // Check if we can apply transformation
            if (!chunk_data.empty()) {
                bool first_is_upper = IsUpperASCII(chunk_data[0]);
                bool all_upper = true;
                for (auto c : chunk_data) {
                    if (IsLowerASCII(c)) {
                        all_upper = false;
                        break;
                    }
                }
                
                if (all_upper && chunk_data.size() > 1) {
                    transform = TRANSFORM_ALL_UPPER;
                } else if (first_is_upper) {
                    transform = TRANSFORM_FIRST_UPPER;
                }
                
                ApplyTransform(chunk_data, transform);
            }
            
        } else if (type == REGION_WIKI_HEADER) {
            // Extract content between == ==
            chunk_data.assign(buffer.begin() + pos + 2, buffer.begin() + region_end - 2);
            
            if (!chunk_data.empty()) {
                bool first_is_upper = IsUpperASCII(chunk_data[0]);
                bool all_upper = true;
                for (auto c : chunk_data) {
                    if (IsLowerASCII(c)) {
                        all_upper = false;
                        break;
                    }
                }
                
                if (all_upper && chunk_data.size() > 1) {
                    transform = TRANSFORM_ALL_UPPER;
                } else if (first_is_upper) {
                    transform = TRANSFORM_FIRST_UPPER;
                }
                
                ApplyTransform(chunk_data, transform);
            }
            
        } else if (type == REGION_XML_TAG) {
            // Extract tag and content separately
            size_t tag_start = pos + 1;
            size_t tag_end = tag_start;
            while (tag_end < buffer.size() && IsLowerASCII(buffer[tag_end])) tag_end++;
            
            size_t tag_len = tag_end - tag_start;
            size_t content_start = tag_end + 1;
            size_t content_end = region_end - tag_len - 3; // </tag>
            
            // Store tag length (1 byte) + tag name + content
            chunk_data.push_back((unsigned char)tag_len);
            chunk_data.insert(chunk_data.end(), buffer.begin() + tag_start, buffer.begin() + tag_end);
            
            std::vector<unsigned char> content(buffer.begin() + content_start, buffer.begin() + content_end);
            
            if (!content.empty()) {
                bool first_is_upper = IsUpperASCII(content[0]);
                bool all_upper = true;
                for (auto c : content) {
                    if (IsLowerASCII(c)) {
                        all_upper = false;
                        break;
                    }
                }
                
                if (all_upper && content.size() > 1) {
                    transform = TRANSFORM_ALL_UPPER;
                } else if (first_is_upper) {
                    transform = TRANSFORM_FIRST_UPPER;
                }
                
                ApplyTransform(content, transform);
            }
            
            chunk_data.insert(chunk_data.end(), content.begin(), content.end());        } else if (type == REGION_UTF8_COMPRESSED) {
            // Collect UTF-8 characters and find range
            uint32_t min_cp = 0xFFFFFFFF;
            uint32_t max_cp = 0;
            std::vector<uint32_t> codepoints;
            
            size_t i = pos;
            while (i < region_end) {
                size_t consumed = 0;
                uint32_t cp = DecodeUTF8(buffer.data() + i, region_end - i, &consumed);
                if (consumed == 0) break;
                
                codepoints.push_back(cp);
                if (cp < min_cp) min_cp = cp;
                if (cp > max_cp) max_cp = cp;
                i += consumed;
            }
            
            // Encode: min_cp (UTF-8) + deltas (1 byte each)
            unsigned char utf8_min[4];
            size_t min_len = EncodeUTF8(min_cp, utf8_min);
            chunk_data.assign(utf8_min, utf8_min + min_len);
            
            for (uint32_t cp : codepoints) {
                chunk_data.push_back((unsigned char)(cp - min_cp));
            }
            
        } else {
            // RAW: copy as-is (region_end already set by DetectRegion)
            chunk_data.assign(buffer.begin() + pos, buffer.begin() + region_end);
        }
        
        // Safety guards
        // 1) XML content must be pure text (no '<')
        if (type == REGION_XML_TAG) {
            if (std::find(chunk_data.begin(), chunk_data.end(), '<') != chunk_data.end()) {
                type = REGION_RAW;
                transform = TRANSFORM_NONE;
                region_end = std::min(pos + 255, file_size);
                chunk_data.assign(buffer.begin() + pos, buffer.begin() + region_end);
            }
        }

        // 2) Do not emit chunks larger than 65535 (header limit) — fallback to RAW
        if (chunk_data.size() > 65535) {
            type = REGION_RAW;
            transform = TRANSFORM_NONE;
            region_end = std::min(pos + 255, file_size);
            chunk_data.assign(buffer.begin() + pos, buffer.begin() + region_end);
        }

        // 3) If transformation makes it bigger, fall back to RAW
        size_t original_size = region_end - pos;
        if (chunk_data.size() > original_size) {
            type = REGION_RAW;
            transform = TRANSFORM_NONE;
            chunk_data.assign(buffer.begin() + pos, buffer.begin() + region_end);
        }
        
        // Write chunk header
        ChunkHeader header;
        header.type = type;
        header.transform = transform;
        header.size = (uint16_t)chunk_data.size();
        header.big_size = (header.size > 255);
        header.context_dependent = false;
        
        WriteChunkHeader(output, header);
        
        // Write chunk data
        fwrite(chunk_data.data(), 1, chunk_data.size(), output);
        
        pos = region_end;
    }
    
    return true;
}

bool Decompress(FILE* input, FILE* output) {
    if (!input || !output) return false;
    
    while (true) {
        ChunkHeader header;
        if (!ReadChunkHeader(input, header)) {
            break; // EOF
        }
        
        // Read chunk data
        std::vector<unsigned char> chunk_data(header.size);
        if (fread(chunk_data.data(), 1, header.size, input) != header.size) {
            fprintf(stderr, "Error: Failed to read chunk data (expected %u bytes)\n", header.size);
            return false;
        }
        
        std::vector<unsigned char> output_data;
        
        if (header.type == REGION_ASCII_SENTENCE) {
            ReverseTransform(chunk_data, header.transform);
            output_data = chunk_data;
            output_data.push_back('.'); // Add back the dot
            
        } else if (header.type == REGION_BRACKETS) {
            output_data.push_back('[');
            output_data.push_back('[');
            ReverseTransform(chunk_data, header.transform);
            output_data.insert(output_data.end(), chunk_data.begin(), chunk_data.end());
            output_data.push_back(']');
            output_data.push_back(']');
            
        } else if (header.type == REGION_WIKI_HEADER) {
            output_data.push_back('=');
            output_data.push_back('=');
            ReverseTransform(chunk_data, header.transform);
            output_data.insert(output_data.end(), chunk_data.begin(), chunk_data.end());
            output_data.push_back('=');
            output_data.push_back('=');
            
        } else if (header.type == REGION_CURLY_BRACKETS) {
            output_data.push_back('{');
            output_data.push_back('{');
            ReverseTransform(chunk_data, header.transform);
            output_data.insert(output_data.end(), chunk_data.begin(), chunk_data.end());
            output_data.push_back('}');
            output_data.push_back('}');
            
        } else if (header.type == REGION_XML_TAG) {
            // Read tag length prefix
            if (chunk_data.empty()) {
                fprintf(stderr, "Error: XML_TAG empty (size=%u)\n", header.size);
                return false;
            }
            
            uint8_t tag_len = chunk_data[0];
            if (1 + tag_len > chunk_data.size()) {
                fprintf(stderr, "Error: XML_TAG tag length %u exceeds chunk size %u\n", (unsigned)tag_len, header.size);
                return false;
            }
            
            std::vector<unsigned char> tag_name(chunk_data.begin() + 1, chunk_data.begin() + 1 + tag_len);
            std::vector<unsigned char> content(chunk_data.begin() + 1 + tag_len, chunk_data.end());
            
            ReverseTransform(content, header.transform);
            
            output_data.push_back('<');
            output_data.insert(output_data.end(), tag_name.begin(), tag_name.end());
            output_data.push_back('>');
            output_data.insert(output_data.end(), content.begin(), content.end());
            output_data.push_back('<');
            output_data.push_back('/');
            output_data.insert(output_data.end(), tag_name.begin(), tag_name.end());
            output_data.push_back('>');
            
        } else if (header.type == REGION_UTF8_COMPRESSED) {
            // Decode min_cp
            size_t consumed = 0;
            uint32_t min_cp = DecodeUTF8(chunk_data.data(), chunk_data.size(), &consumed);
            if (consumed == 0) {
                fprintf(stderr, "Error: UTF8_COMPRESSED failed to decode min_cp (size=%u)\n", header.size);
                return false;
            }
            
            // Decode deltas
            for (size_t i = consumed; i < chunk_data.size(); ++i) {
                uint32_t cp = min_cp + chunk_data[i];
                unsigned char utf8_bytes[4];
                size_t len = EncodeUTF8(cp, utf8_bytes);
                output_data.insert(output_data.end(), utf8_bytes, utf8_bytes + len);
            }
            
        } else {
            // RAW
            output_data = chunk_data;
        }
        
        fwrite(output_data.data(), 1, output_data.size(), output);
    }
    return true;
}

} // namespace chunked_preprocess











