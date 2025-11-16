#include "utf8_preprocessor.h"
#include <algorithm>
#include <unordered_map>
#include <cstring>

namespace utf8_preprocess {

// UTF-8 decoder
static uint32_t DecodeUTF8(const unsigned char* data, size_t remaining, size_t* bytes_consumed) {
    if (remaining == 0) { *bytes_consumed = 0; return 0; }
    unsigned char b0 = data[0];
    if ((b0 & 0x80) == 0) { *bytes_consumed = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (remaining < 2) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1];
        if ((b1 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 2;
        return ((b0 & 0x1F) << 6) | (b1 & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (remaining < 3) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1], b2 = data[2];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 3;
        return ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (remaining < 4) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1], b2 = data[2], b3 = data[3];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        *bytes_consumed = 4;
        return ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
    }
    *bytes_consumed = 0; return 0;
}

static bool IsUnicodeSpace(uint32_t cp) {
    if (cp == 0x00A0 || cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return true;
    if (cp >= 0x2000 && cp <= 0x200B) return true;
    return false;
}

static bool IsSeparator(uint32_t cp) {
    if (cp <= 0x20) return true;
    switch (cp) {
        case '!': case '"': case '#': case '$': case '%': case '&': case '\'': case '(': case ')': case '*': case '+':
        case ',': case '-': case '.': case '/': case ':': case ';': case '<': case '=': case '>': case '?': case '@':
        case '[': case '\\': case ']': case '^': case '_': case '`': case '{': case '|': case '}': case '~':
            return true;
        default: break;
    }
    if (IsUnicodeSpace(cp)) return true;
    return false;
}

// Trie for word collection
struct TrieNode {
    int parent = -1;
    unsigned char in_byte = 0;
    uint64_t count = 0;
    uint32_t length_bytes = 0;
    bool terminal = false;
};

struct EdgeKeyHash {
    size_t operator()(const uint64_t& k) const noexcept { 
        return (size_t)((k >> 32) ^ (k & 0xFFFFFFFFu)); 
    }
};

struct WordEntry {
    std::string word;
    uint64_t count;
    uint32_t length;
    uint64_t score;
};

// Analyze input and collect unused ASCII slots + top words
static bool AnalyzeInput(const std::vector<unsigned char>& buffer,
                         std::vector<unsigned char>& unused_slots,
                         std::vector<WordEntry>& top_words) {
    size_t file_size = buffer.size();
    
    // Track used ASCII bytes
    bool used_ascii[128] = {false};
    
    // First pass: identify used ASCII
    size_t pos = 0;
    while (pos < file_size) {
        size_t consumed = 0;
        uint32_t cp = DecodeUTF8(buffer.data() + pos, file_size - pos, &consumed);
        if (consumed == 0) {
            pos++;
            continue;
        }
        if (cp < 128) {
            used_ascii[cp] = true;
        }
        pos += consumed;
    }
    
    // Collect unused ASCII slots
    for (int i = 0; i < 128; ++i) {
        if (!used_ascii[i]) {
            unused_slots.push_back((unsigned char)i);
        }
    }
    
    if (unused_slots.empty()) {
        return true; // No optimization possible
    }
    
    // Build trie for word collection
    std::vector<TrieNode> nodes;
    nodes.reserve(1 << 20);
    nodes.push_back(TrieNode{}); // root
    
    std::unordered_map<uint64_t, int, EdgeKeyHash> edge_map;
    edge_map.reserve(1 << 20);
    
    auto get_child = [&](int parent, unsigned char b) -> int {
        uint64_t key = ((uint64_t)parent << 8) | (uint64_t)b;
        auto it = edge_map.find(key);
        if (it != edge_map.end()) return it->second;
        int idx = (int)nodes.size();
        TrieNode n; 
        n.parent = parent; 
        n.in_byte = b;
        nodes.push_back(n);
        edge_map.emplace(key, idx);
        return idx;
    };
    
    pos = 0;
    int current = 0;
    uint32_t current_len = 0;
    
    auto end_word = [&]() {
        if (current != 0 && current_len > 0) {
            TrieNode& n = nodes[current];
            n.terminal = true;
            n.count += 1;
            if (n.length_bytes == 0) n.length_bytes = current_len;
        }
        current = 0;
        current_len = 0;
    };
    
    // Second pass: build word trie
    while (pos < file_size) {
        size_t consumed = 0;
        uint32_t cp = DecodeUTF8(buffer.data() + pos, file_size - pos, &consumed);
        if (consumed == 0) {
            end_word();
            pos++;
            continue;
        }
        if (IsSeparator(cp)) {
            end_word();
        } else {
            for (size_t i = 0; i < consumed; ++i) {
                current = get_child(current, buffer[pos + i]);
            }
            current_len += (uint32_t)consumed;
        }
        pos += consumed;
    }
    end_word();
    
    // Collect terminal words and score them
    std::vector<WordEntry> entries;
    for (int i = 1; i < (int)nodes.size(); ++i) {
        if (nodes[i].terminal) {
            uint64_t cnt = nodes[i].count;
            uint32_t len = nodes[i].length_bytes;
            if (len > 1 && cnt > 0) { // Only words longer than 1 byte benefit
                uint64_t score = (uint64_t)(len - 1) * cnt;
                
                // Reconstruct word
                std::string word;
                int cur = i;
                while (cur != 0) {
                    word.push_back((char)nodes[cur].in_byte);
                    cur = nodes[cur].parent;
                }
                std::reverse(word.begin(), word.end());
                
                entries.push_back({word, cnt, len, score});
            }
        }
    }
    
    // Sort by score descending
    std::sort(entries.begin(), entries.end(), [](const WordEntry& a, const WordEntry& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.count != b.count) return a.count > b.count;
        return a.length > b.length;
    });
    
    // Take top N words (limited by available slots)
    size_t max_words = std::min(unused_slots.size(), entries.size());
    top_words.assign(entries.begin(), entries.begin() + max_words);
    
    return true;
}

bool Compress(FILE* input, FILE* output) {
    if (!input || !output) return false;
    
    // Read entire input
    std::fseek(input, 0, SEEK_END);
    size_t file_size = std::ftell(input);
    std::fseek(input, 0, SEEK_SET);
    
    std::vector<unsigned char> buffer(file_size);
    if (std::fread(buffer.data(), 1, file_size, input) != file_size) {
        return false;
    }
    
    // Analyze input
    std::vector<unsigned char> unused_slots;
    std::vector<WordEntry> top_words;
    
    if (!AnalyzeInput(buffer, unused_slots, top_words)) {
        return false;
    }
    
    // Write header
    unsigned char slot_count = (unsigned char)std::min(top_words.size(), (size_t)255);
    std::fwrite(&slot_count, 1, 1, output);
    
    if (slot_count == 0) {
        // No optimization - copy input as-is
        std::fwrite(buffer.data(), 1, file_size, output);
        return true;
    }
    
    // Build replacement map: word -> slot_byte
    std::unordered_map<std::string, unsigned char> word_to_slot;
    for (size_t i = 0; i < slot_count; ++i) {
        word_to_slot[top_words[i].word] = unused_slots[i];
    }
    
    // Write mapping table: for each slot, write slot_byte, word_length, word_bytes
    for (size_t i = 0; i < slot_count; ++i) {
        unsigned char slot_byte = unused_slots[i];
        const std::string& word = top_words[i].word;
        unsigned char word_len = (unsigned char)word.size();
        
        std::fwrite(&slot_byte, 1, 1, output);
        std::fwrite(&word_len, 1, 1, output);
        std::fwrite(word.data(), 1, word_len, output);
    }
    
    // Transform input: replace words with slots
    size_t pos = 0;
    std::vector<unsigned char> result;
    result.reserve(file_size);
    
    while (pos < file_size) {
        bool matched = false;
        
        // Try to match words starting at current position
        for (const auto& kv : word_to_slot) {
            const std::string& word = kv.first;
            unsigned char slot = kv.second;
            
            if (pos + word.size() <= file_size) {
                if (std::memcmp(buffer.data() + pos, word.data(), word.size()) == 0) {
                    result.push_back(slot);
                    pos += word.size();
                    matched = true;
                    break;
                }
            }
        }
        
        if (!matched) {
            result.push_back(buffer[pos]);
            pos++;
        }
    }
    
    // Write transformed data
    std::fwrite(result.data(), 1, result.size(), output);
    
    return true;
}

bool Decompress(FILE* input, FILE* output) {
    if (!input || !output) return false;
    
    // Read slot count
    unsigned char slot_count = 0;
    if (std::fread(&slot_count, 1, 1, input) != 1) {
        return false;
    }
    
    if (slot_count == 0) {
        // No transformation - copy rest of input
        unsigned char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), input)) > 0) {
            std::fwrite(buf, 1, n, output);
        }
        return true;
    }
    
    // Read mapping table: slot_byte -> word
    std::unordered_map<unsigned char, std::string> slot_to_word;
    
    for (size_t i = 0; i < slot_count; ++i) {
        unsigned char slot_byte = 0;
        unsigned char word_len = 0;
        
        if (std::fread(&slot_byte, 1, 1, input) != 1) return false;
        if (std::fread(&word_len, 1, 1, input) != 1) return false;
        
        std::string word(word_len, '\0');
        if (std::fread(&word[0], 1, word_len, input) != word_len) return false;
        
        slot_to_word[slot_byte] = word;
    }
    
    // Read and transform data
    unsigned char byte;
    while (std::fread(&byte, 1, 1, input) == 1) {
        auto it = slot_to_word.find(byte);
        if (it != slot_to_word.end()) {
            // Replace slot with original word
            std::fwrite(it->second.data(), 1, it->second.size(), output);
        } else {
            // Copy byte as-is
            std::fwrite(&byte, 1, 1, output);
        }
    }
    
    return true;
}

} // namespace utf8_preprocess
