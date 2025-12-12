#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#undef min
#undef max
#endif

// UTF-8 decode: returns codepoint and sets bytes_consumed; returns 0 with bytes_consumed=0 for invalid
static uint32_t DecodeUTF8(const unsigned char* data, size_t remaining, size_t* bytes_consumed) {
    if (remaining == 0) { *bytes_consumed = 0; return 0; }
    unsigned char b0 = data[0];
    if ((b0 & 0x80) == 0) { *bytes_consumed = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (remaining < 2) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1];
        if ((b1 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        uint32_t cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
        *bytes_consumed = 2; return cp;
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (remaining < 3) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1], b2 = data[2];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        uint32_t cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
        *bytes_consumed = 3; return cp;
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (remaining < 4) { *bytes_consumed = 0; return 0; }
        unsigned char b1 = data[1], b2 = data[2], b3 = data[3];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) { *bytes_consumed = 0; return 0; }
        uint32_t cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
        *bytes_consumed = 4; return cp;
    }
    *bytes_consumed = 0; return 0;
}

static bool IsUnicodeSpace(uint32_t cp) {
    // Common Unicode spaces including NBSP and various em/en spaces
    if (cp == 0x00A0 || cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return true;
    if (cp >= 0x2000 && cp <= 0x200B) return true;
    return false;
}

static bool IsSeparator(uint32_t cp) {
    // ASCII controls and space
    if (cp <= 0x20) return true;
    // ASCII punctuation (treat underscore as separator as well)
    switch (cp) {
        case '!' : case '"': case '#': case '$': case '%': case '&': case '\'': case '(': case ')': case '*': case '+':
        case ',': case '-': case '.': case '/': case ':': case ';': case '<': case '=': case '>': case '?': case '@':
        case '[': case '\\': case ']': case '^': case '_': case '`': case '{': case '|': case '}': case '~':
            return true;
        default: break;
    }
    if (IsUnicodeSpace(cp)) return true;
    return false;
}

// Compact trie using global edge hash (parent_index, byte) -> child_index
struct TrieNode {
    int parent = -1;
    unsigned char in_byte = 0;
    uint64_t count = 0;          // word occurrence count at terminal
    uint32_t length_bytes = 0;   // word length in bytes (UTF-8) at terminal
    bool terminal = false;
};

struct EdgeKeyHash {
    size_t operator()(const uint64_t& k) const noexcept { return (size_t)((k >> 32) ^ (k & 0xFFFFFFFFu)); }
};

int main(int argc, char** argv) {
#ifdef _WIN32
    // Initialize console to support Unicode (best effort)
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "Error: cannot open %s\n", path); return 1; }

    std::fseek(f, 0, SEEK_END);
    size_t file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    std::vector<unsigned char> buffer(file_size);
    size_t read = std::fread(buffer.data(), 1, file_size, f);
    std::fclose(f);
    if (read != file_size) { std::fprintf(stderr, "Error: failed to read entire file\n"); return 1; }

    // Trie storage
    std::vector<TrieNode> nodes;
    nodes.reserve(1 << 20);
    nodes.push_back(TrieNode{}); // root index 0

    std::unordered_map<uint64_t, int, EdgeKeyHash> edge_map;
    edge_map.reserve(1 << 20);

    auto get_child = [&](int parent, unsigned char b) -> int {
        uint64_t key = ( (uint64_t)parent << 8 ) | (uint64_t)b;
        auto it = edge_map.find(key);
        if (it != edge_map.end()) return it->second;
        // create
        int idx = (int)nodes.size();
        TrieNode n; n.parent = parent; n.in_byte = b;
        nodes.push_back(n);
        edge_map.emplace(key, idx);
        return idx;
    };

    size_t pos = 0;
    int current = 0; // current node in trie
    uint32_t current_len_bytes = 0; // bytes length of current word

    auto end_word = [&]() {
        if (current != 0 && current_len_bytes > 0) {
            TrieNode& n = nodes[current];
            n.terminal = true;
            n.count += 1;
            if (n.length_bytes == 0) n.length_bytes = current_len_bytes;
        }
        current = 0;
        current_len_bytes = 0;
    };

    while (pos < file_size) {
        size_t consumed = 0;
        uint32_t cp = DecodeUTF8(buffer.data() + pos, file_size - pos, &consumed);
        if (consumed == 0) {
            // treat invalid byte as separator boundary
            end_word();
            pos += 1;
            continue;
        }
        if (IsSeparator(cp)) {
            end_word();
        } else {
            // append original UTF-8 bytes to trie path
            for (size_t i = 0; i < consumed; ++i) {
                current = get_child(current, buffer[pos + i]);
            }
            current_len_bytes += (uint32_t)consumed;
        }
        pos += consumed;
    }
    // finalize last
    end_word();

    // Collect terminal words
    struct WordEntry { uint64_t count; uint32_t len; uint64_t score; int node; };
    std::vector<WordEntry> entries;
    entries.reserve(1 << 20);
    for (int i = 1; i < (int)nodes.size(); ++i) {
        if (nodes[i].terminal) {
            uint64_t cnt = nodes[i].count;
            uint32_t len = nodes[i].length_bytes;
            if (len >= 1 && cnt > 0) {
                uint64_t score = (uint64_t)(len > 0 ? (len - 1) : 0) * cnt;
                entries.push_back({cnt, len, score, i});
            }
        }
    }

    std::sort(entries.begin(), entries.end(), [](const WordEntry& a, const WordEntry& b){
        if (a.score != b.score) return a.score > b.score;
        if (a.count != b.count) return a.count > b.count;
        return a.len > b.len;
    });

    // Print top 31 words and compute potential gain
    size_t topN = std::min<size_t>(31, entries.size());
    std::printf("Top %zu words by score ((bytes-1) * count)\n", topN);
    std::printf("====================================================================\n");
    std::printf("Rank  Bytes  Count        Score   Word (UTF-8)\n");
    std::printf("----  -----  -----------  -------  --------------------------------\n");

    auto reconstruct_word = [&](int node)->std::string {
        std::string s;
        int cur = node;
        while (cur != 0) {
            s.push_back((char)nodes[cur].in_byte);
            cur = nodes[cur].parent;
        }
        std::reverse(s.begin(), s.end());
        return s;
    };

    unsigned long long saved = 0ULL;
    for (size_t i = 0; i < topN; ++i) {
        const auto& e = entries[i];
        saved += (unsigned long long)e.count * (unsigned long long)(e.len - 1);
        std::string w = reconstruct_word(e.node);
        std::printf("%4zu  %5u  %11llu  %7llu  ", i+1, e.len, (unsigned long long)e.count, (unsigned long long)e.score);
        // print as UTF-8 raw; Windows terminal may need UTF-8 codepage
        std::fwrite(w.data(), 1, w.size(), stdout);
        std::printf("\n");
    }

    unsigned long long original = (unsigned long long)file_size;
    unsigned long long transformed = (saved > original) ? 0ULL : (original - saved);
    double gain = original ? (100.0 * saved / (double)original) : 0.0;

    std::printf("\nPotential remap statistics (replace top %zu words with 1 byte):\n", topN);
    std::printf("  Total bytes saved (ideal): %llu\n", saved);
    std::printf("  Original size            : %llu\n", original);
    std::printf("  Transformed size (ideal) : %llu\n", transformed);
    std::printf("  Gain                     : %.2f%%\n", gain);
    std::printf("  Note: Does not include mapping table overhead.\n");


    // Reconstruct dictionary for lowercase (ASCII) words only, merging counts for case variants
    // and output sorted by score descending to dictionary\words.dic (UTF-8 lines).
    struct AggEntry { unsigned long long count = 0ULL; uint32_t len = 0; };
    auto to_ascii_lower = [](const std::string& s) {
        std::string out; out.reserve(s.size());
        for (unsigned char c : s) {
            if (c >= 'A' && c <= 'Z') out.push_back((char)(c - 'A' + 'a'));
            else out.push_back((char)c);
        }
        return out;
    };

    std::unordered_map<std::string, AggEntry> agg;
    agg.reserve(entries.size());
    for (const auto& e : entries) {
        std::string w = reconstruct_word(e.node);
        std::string lw = to_ascii_lower(w);
        AggEntry& a = agg[lw];
        a.count += (unsigned long long)e.count;
        // word length in bytes remains the same for ASCII case changes and general UTF-8 preservation
        if (a.len == 0) a.len = e.len; // set once; all variants have same byte length along the same path
    }

    struct DictItem { std::string word; unsigned long long count; uint32_t len; unsigned long long score; };
    std::vector<DictItem> dict;
    dict.reserve(agg.size());
    for (const auto& kv : agg) {
        const std::string& w = kv.first;
        const AggEntry& a = kv.second;
        uint32_t len = a.len ? a.len : (uint32_t)w.size();
        unsigned long long score = (unsigned long long)((len > 0) ? (len - 1) : 0) * a.count;
        dict.push_back(DictItem{w, a.count, len, score});
    }

    std::sort(dict.begin(), dict.end(), [](const DictItem& x, const DictItem& y){
        if (x.score != y.score) return x.score > y.score;
        if (x.count != y.count) return x.count > y.count;
        if (x.len != y.len) return x.len > y.len;
        return x.word < y.word;
    });

    // Prune dictionary by score drop-off heuristic to keep impactful entries
    // Heuristic:
    //  - Always keep at least min_keep entries
    //  - Stop when marginal score falls below min_score OR
    //    when drop ratio (prev_score / cur_score) exceeds drop_ratio after warmup
    //  - Also stop when cumulative saved reaches target_coverage of original size
    const size_t min_keep = 256;              // ensure a base dictionary size
    const unsigned long long min_score = 2;   // require at least 2 bytes saved total
    const double drop_ratio = 8.0;            // large cliff in score suggests tail
    const size_t warmup = 512;                // wait before applying drop detection
    const double target_coverage = 0.95;      // aim to cover 95% of ideal savings for kept items
    const size_t hard_cap = 65535;            // absolute max entries to keep

    unsigned long long total_possible = 0ULL;
    for (const auto& item : dict) total_possible += item.score;

    std::vector<DictItem> pruned;
    pruned.reserve(std::min(hard_cap, dict.size()));
    unsigned long long cum = 0ULL;
    unsigned long long prev_score = dict.empty() ? 0ULL : dict.front().score;
    for (size_t i = 0; i < dict.size(); ++i) {
        const auto& item = dict[i];
        if (i < min_keep) {
            pruned.push_back(item);
            cum += item.score;
            prev_score = item.score;
            continue;
        }
        // Coverage target reached
        if (total_possible > 0ULL) {
            double cov = (double)cum / (double)total_possible;
            if (cov >= target_coverage) break;
        }
        // Hard cap
        if (pruned.size() >= hard_cap) break;
        // Minimum score requirement
        if (item.score < min_score) break;
        // Drop-off detection after warmup
        if (i >= warmup && prev_score > 0ULL) {
            double ratio = (double)prev_score / (double)item.score;
            if (ratio >= drop_ratio) break;
        }
        pruned.push_back(item);
        cum += item.score;
        prev_score = item.score;
    }

    // Write dictionary to file (one UTF-8 word per line)
    {
        const char* out_path = "dictionary\\words.dic";
        FILE* df = std::fopen(out_path, "wb");
        if (!df) {
            std::fprintf(stderr, "Warning: cannot open %s for writing.\n", out_path);
        } else {
            for (const auto& item : pruned) {
                std::fwrite(item.word.data(), 1, item.word.size(), df);
                std::fputc('\n', df);
            }
            std::fclose(df);
            std::printf("\nWrote %zu pruned entries to %s\n", pruned.size(), out_path);
        }
    }



    return 0;
}
