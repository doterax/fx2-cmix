// dictionary_compress.cpp — High-byte dictionary preprocessor for cmix
//
// Uses the same encoding scheme as cmix's original preprocessor (0x80+ codes),
// but with frequency-optimal dictionary building, aggressive pruning of words
// that don't improve compression, and substring (prefix/suffix) matching.
//
// Encoding scheme (compatible with cmix decoder):
//   - Words mapped to 1/2/3-byte codes in 0x80+ range
//   - Tier 1:  indices 0..79       -> 1 byte  (0x80 + idx)
//   - Tier 2:  indices 80..3919    -> 2 bytes  (0xD0 + hi, 0x80 + lo)  hi=48, lo=80
//   - Tier 3:  indices 3920..44879 -> 3 bytes  (0xF0 + hi, 0xD0 + mid, 0x80 + lo)
//   - Case markers: 0x40 = capitalized, 0x07 = all-uppercase, 0x06 = end-upper
//   - Escape: 0x0C prefix for literal bytes that collide with markers/high bytes
//
// Build modes:
//   build-dict  -- Build optimal dictionary from a corpus file
//   compress    -- Preprocess input using a dictionary (produces binary output)
//   decompress  -- Reverse the preprocessing
//   analyze     -- Show compression statistics without writing output
//   verify      -- Roundtrip verification

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Constants -- same as cmix's dictionary.cpp
// ---------------------------------------------------------------------------
static const unsigned char kCapitalized = 0x40;
static const unsigned char kUppercase   = 0x07;
static const unsigned char kEndUpper    = 0x06;
static const unsigned char kEscape      = 0x0C;

static const int kBoundary1 = 80;
static const int kBoundary2 = kBoundary1 + 48 * 80;          // 3920
static const int kBoundary3 = kBoundary2 + 16 * 32 * 80;     // 44880
static const int kMaxDictSize = kBoundary3;                   // 44880 words max

// ---------------------------------------------------------------------------
// High-byte code encoding/decoding (matches cmix format exactly)
// ---------------------------------------------------------------------------
static unsigned int index_to_bytes(int idx) {
  if (idx < kBoundary1) {
    return 0x80 + idx;
  } else if (idx < kBoundary2) {
    int adj = idx - kBoundary1;
    unsigned int bytes = 0xD0 + (adj / 80);
    bytes += (0x80 + (adj % 80)) << 8;
    return bytes;
  } else if (idx < kBoundary3) {
    int adj = idx - kBoundary2;
    unsigned int bytes = 0xF0 + ((adj / 80) / 32);
    bytes += (0xD0 + ((adj / 80) % 32)) << 8;
    bytes += (0x80 + (adj % 80)) << 16;
    return bytes;
  }
  return 0;
}

static int code_byte_length(int idx) {
  if (idx < kBoundary1) return 1;
  if (idx < kBoundary2) return 2;
  if (idx < kBoundary3) return 3;
  return 0;
}

static void emit_code_bytes(unsigned int bytes, std::string &out) {
  out.push_back((unsigned char)(bytes & 0xFF));
  if (bytes & 0xFF00)
    out.push_back((unsigned char)((bytes >> 8) & 0xFF));
  if (bytes & 0xFF0000)
    out.push_back((unsigned char)((bytes >> 16) & 0xFF));
}

static void emit_literal_byte(unsigned char c, std::string &out) {
  if (c == kEndUpper || c == kEscape || c == kUppercase ||
      c == kCapitalized || c >= 0x80) {
    out.push_back(kEscape);
  }
  out.push_back(c);
}

// ---------------------------------------------------------------------------
// Dictionary: maps words <-> indices
// ---------------------------------------------------------------------------
struct DictCodec {
  std::unordered_map<std::string, int> word_to_idx;
  std::vector<std::string>             idx_to_word;
  unsigned int                         longest_word = 0;

  int size() const { return (int)idx_to_word.size(); }

  bool load_cmix_format(const std::string &path) {
    word_to_idx.clear();
    idx_to_word.clear();
    longest_word = 0;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
      std::string word;
      for (char c : line) {
        if (c >= 'a' && c <= 'z') word += c;
        else if (!word.empty()) break;
      }
      if (word.empty()) continue;
      if ((int)idx_to_word.size() >= kMaxDictSize) break;
      if (word.size() > longest_word) longest_word = (unsigned int)word.size();
      int idx = (int)idx_to_word.size();
      word_to_idx[word] = idx;
      idx_to_word.push_back(word);
    }
    return !idx_to_word.empty();
  }

  bool save_cmix_format(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    for (const auto &w : idx_to_word)
      out << w << "\n";
    return true;
  }
};

// ---------------------------------------------------------------------------
// Encoder: transforms text -> binary using dictionary (high-byte codes)
// ---------------------------------------------------------------------------
class HBEncoder {
public:
  HBEncoder(const DictCodec &dict) : dict_(dict) {}

  std::string encode(const std::string &input) const {
    std::string out;
    out.reserve(input.size());

    std::string word;
    int num_upper = 0, num_lower = 0;
    int len = (int)input.size();

    for (int pos = 0; pos < len; ++pos) {
      unsigned char c = input[pos];
      bool advance = false;

      if (word.size() > dict_.longest_word) {
        advance = true;
      } else if (c >= 'a' && c <= 'z') {
        if (num_upper > 1) {
          advance = true;
        } else {
          ++num_lower;
          word += c;
        }
      } else if (c >= 'A' && c <= 'Z') {
        if (num_lower > 0) {
          advance = true;
        } else {
          ++num_upper;
          word += (char)(c - 'A' + 'a');
        }
      } else {
        advance = true;
      }

      if (pos == len - 1 && !advance) {
        encode_word(word, num_upper, false, out);
      }
      if (advance) {
        if (word.empty()) {
          emit_literal_byte(c, out);
        } else {
          bool next_lower = (c >= 'a' && c <= 'z');
          encode_word(word, num_upper, next_lower, out);
          num_lower = 0;
          num_upper = 0;
          word.clear();
          if (next_lower) {
            ++num_lower;
            word += c;
          } else if (c >= 'A' && c <= 'Z') {
            ++num_upper;
            word += (char)(c - 'A' + 'a');
          } else {
            emit_literal_byte(c, out);
          }
          if (pos == len - 1 && !word.empty()) {
            encode_word(word, num_upper, false, out);
          }
        }
      }
    }
    return out;
  }

  struct Stats {
    int64_t total_words_seen  = 0;
    int64_t words_encoded     = 0;
    int64_t words_substring   = 0;
    int64_t bytes_raw         = 0;
    int64_t bytes_encoded     = 0;
  };

  Stats compute_stats(const std::string &input) const {
    Stats st;
    st.bytes_raw = (int64_t)input.size();
    std::string encoded = encode(input);
    st.bytes_encoded = (int64_t)encoded.size();

    std::string word;
    int num_upper = 0, num_lower = 0;
    int len = (int)input.size();
    for (int pos = 0; pos < len; ++pos) {
      unsigned char c = input[pos];
      bool advance = false;
      if (word.size() > dict_.longest_word) advance = true;
      else if (c >= 'a' && c <= 'z') {
        if (num_upper > 1) advance = true;
        else { ++num_lower; word += c; }
      } else if (c >= 'A' && c <= 'Z') {
        if (num_lower > 0) advance = true;
        else { ++num_upper; word += (char)(c - 'A' + 'a'); }
      } else advance = true;

      auto flush_word = [&]() {
        if (word.empty()) return;
        st.total_words_seen++;
        auto it = dict_.word_to_idx.find(word);
        if (it != dict_.word_to_idx.end()) st.words_encoded++;
        word.clear(); num_upper = 0; num_lower = 0;
      };

      if (pos == len - 1 && !advance) flush_word();
      if (advance) {
        flush_word();
        if (c >= 'a' && c <= 'z') { word += c; num_lower = 1; }
        else if (c >= 'A' && c <= 'Z') { word += (char)(c-'A'+'a'); num_upper = 1; }
        if (pos == len - 1 && !word.empty()) flush_word();
      }
    }
    return st;
  }

private:
  const DictCodec &dict_;

  void encode_word(const std::string &word, int num_upper, bool next_lower,
                   std::string &out) const {
    if (num_upper > 1) out.push_back(kUppercase);
    else if (num_upper == 1) out.push_back(kCapitalized);

    auto it = dict_.word_to_idx.find(word);
    if (it != dict_.word_to_idx.end()) {
      emit_code_bytes(index_to_bytes(it->second), out);
    } else if (!encode_substring(word, out)) {
      for (char c : word)
        out.push_back((unsigned char)c);
    }

    if (num_upper > 1 && next_lower)
      out.push_back(kEndUpper);
  }

  bool encode_substring(const std::string &word, std::string &out) const {
    if (word.size() <= 7) return false;
    unsigned int sz = (unsigned int)word.size() - 1;
    if (sz > dict_.longest_word) sz = dict_.longest_word;

    // Try suffix match (emit literal prefix + code for suffix)
    std::string suffix = word.substr(word.size() - sz, sz);
    while (suffix.size() >= 7) {
      auto it = dict_.word_to_idx.find(suffix);
      if (it != dict_.word_to_idx.end()) {
        for (unsigned int i = 0; i < word.size() - suffix.size(); ++i)
          out.push_back((unsigned char)word[i]);
        emit_code_bytes(index_to_bytes(it->second), out);
        return true;
      }
      suffix.erase(0, 1);
    }
    // Try prefix match (code for prefix + literal suffix)
    std::string prefix = word.substr(0, sz);
    while (prefix.size() >= 7) {
      auto it = dict_.word_to_idx.find(prefix);
      if (it != dict_.word_to_idx.end()) {
        emit_code_bytes(index_to_bytes(it->second), out);
        for (unsigned int i = (unsigned int)prefix.size(); i < word.size(); ++i)
          out.push_back((unsigned char)word[i]);
        return true;
      }
      prefix.erase(prefix.size() - 1, 1);
    }
    return false;
  }
};

// ---------------------------------------------------------------------------
// Decoder: transforms binary -> text using dictionary
// ---------------------------------------------------------------------------
class HBDecoder {
public:
  HBDecoder(const DictCodec &dict) : dict_(dict) {}

  std::string decode(const std::string &input) const {
    std::string out;
    out.reserve(input.size() * 2);
    bool decode_upper = false, decode_capital = false;
    size_t pos = 0;

    while (pos < input.size()) {
      unsigned char c = input[pos++];

      if (c == kEscape) {
        decode_upper = false;
        if (pos < input.size())
          out.push_back(input[pos++]);
      } else if (c == kUppercase) {
        decode_upper = true;
      } else if (c == kCapitalized) {
        decode_capital = true;
      } else if (c == kEndUpper) {
        decode_upper = false;
      } else if (c >= 0x80) {
        unsigned int bytes = c;
        if (c > 0xCF && pos < input.size()) {
          unsigned char c2 = input[pos++];
          bytes += (unsigned int)c2 << 8;
          if (c2 > 0xCF && pos < input.size()) {
            unsigned char c3 = input[pos++];
            bytes += (unsigned int)c3 << 16;
          }
        }
        int idx = bytes_to_index(bytes);
        if (idx >= 0 && idx < dict_.size()) {
          const std::string &word = dict_.idx_to_word[idx];
          for (size_t i = 0; i < word.size(); ++i) {
            char ch = word[i];
            if (i == 0 && decode_capital) {
              ch = (ch - 'a') + 'A';
              decode_capital = false;
            }
            if (decode_upper)
              ch = (ch - 'a') + 'A';
            out.push_back(ch);
          }
        }
      } else {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
          decode_upper = false;
        if (decode_capital || decode_upper) {
          if (c >= 'a' && c <= 'z')
            c = (c - 'a') + 'A';
        }
        if (decode_capital) decode_capital = false;
        out.push_back(c);
      }
    }
    return out;
  }

private:
  const DictCodec &dict_;

  static int bytes_to_index(unsigned int bytes) {
    unsigned char b0 = bytes & 0xFF;
    unsigned char b1 = (bytes >> 8) & 0xFF;
    unsigned char b2 = (bytes >> 16) & 0xFF;

    if (b0 >= 0x80 && b0 <= 0xCF) {
      // Tier 1: single byte
      return b0 - 0x80;
    }
    if (b0 >= 0xD0) {
      if (b1 >= 0x80 && b1 <= 0xCF) {
        // Tier 2: two bytes, second byte in 0x80..0xCF
        return kBoundary1 + (b0 - 0xD0) * 80 + (b1 - 0x80);
      }
      if (b0 >= 0xF0 && b1 >= 0xD0 && b2 >= 0x80) {
        // Tier 3: three bytes
        return kBoundary2 + (b0 - 0xF0) * 32 * 80 + (b1 - 0xD0) * 80 + (b2 - 0x80);
      }
    }
    return -1;
  }
};

// ---------------------------------------------------------------------------
// Byte-level entropy computation
// ---------------------------------------------------------------------------
static double byte_entropy(const std::string &data) {
  if (data.empty()) return 0.0;
  uint64_t freq[256] = {};
  for (unsigned char c : data) freq[c]++;
  double ent = 0.0;
  double n = (double)data.size();
  for (int i = 0; i < 256; ++i) {
    if (freq[i] == 0) continue;
    double p = freq[i] / n;
    ent -= p * std::log2(p);
  }
  return ent;
}

static int count_distinct_bytes(const std::string &data) {
  bool seen[256] = {};
  for (unsigned char c : data) seen[c] = true;
  int count = 0;
  for (int i = 0; i < 256; ++i) if (seen[i]) count++;
  return count;
}

// ---------------------------------------------------------------------------
// Dictionary builder: frequency-optimal with pruning
// ---------------------------------------------------------------------------
struct WordStats {
  std::string word;
  uint64_t    count = 0;
};

class DictBuilder {
public:
  void scan(const std::string &data) {
    std::string word;
    int num_upper = 0, num_lower = 0;

    for (size_t i = 0; i < data.size(); ++i) {
      unsigned char c = data[i];
      bool advance = false;

      if (c >= 'a' && c <= 'z') {
        if (num_upper > 1) advance = true;
        else { ++num_lower; word += c; }
      } else if (c >= 'A' && c <= 'Z') {
        if (num_lower > 0) advance = true;
        else { ++num_upper; word += (char)(c - 'A' + 'a'); }
      } else {
        advance = true;
      }

      if (i == data.size() - 1 && !advance) {
        add_word(word);
        word.clear(); num_upper = 0; num_lower = 0;
      }
      if (advance) {
        if (!word.empty()) {
          add_word(word);
          num_lower = 0; num_upper = 0; word.clear();
          if (c >= 'a' && c <= 'z') { ++num_lower; word += c; }
          else if (c >= 'A' && c <= 'Z') { ++num_upper; word += (char)(c-'A'+'a'); }
          if (i == data.size() - 1 && !word.empty()) {
            add_word(word);
            word.clear();
          }
        }
      }
    }
  }

  std::vector<WordStats> get_sorted() const {
    std::vector<WordStats> result;
    result.reserve(freq_.size());
    for (auto &kv : freq_)
      result.push_back({kv.first, kv.second});
    std::sort(result.begin(), result.end(), [](auto &a, auto &b) {
      if (a.count != b.count) return a.count > b.count;
      return a.word < b.word;
    });
    return result;
  }

private:
  std::unordered_map<std::string, uint64_t> freq_;

  void add_word(const std::string &w) {
    if (!w.empty()) freq_[w]++;
  }
};

// Build an optimized dictionary: greedily assign slots to maximize savings
static DictCodec build_optimal_dict(const std::vector<WordStats> &candidates,
                                    int max_entries, bool verbose) {
  DictCodec dict;

  // For each slot i, the code costs code_byte_length(i) bytes.
  // A word W at slot i saves: count(W) * (len(W) - codelen(i)) bytes per occurrence.
  // We greedily assign the word with highest savings to each slot.
  //
  // Since tier boundaries are at 80-3920-44880, we compute savings per tier:

  struct Candidate {
    int         orig_idx;
    std::string word;
    uint64_t    count;
    int64_t     savings; // computed per-tier
  };

  std::vector<Candidate> cands;
  cands.reserve(candidates.size());
  for (int i = 0; i < (int)candidates.size(); ++i) {
    auto &ws = candidates[i];
    if (ws.word.empty() || ws.word.size() <= 1) continue; // 1-char words: min code=1, saves 0
    cands.push_back({i, ws.word, ws.count, 0});
  }

  // Fill slots greedily, tier by tier
  int slot = 0;
  int64_t total_savings = 0;

  auto fill_tier = [&](int tier_end, int code_len) {
    // Compute savings for this tier
    for (auto &c : cands)
      c.savings = (int64_t)c.count * ((int64_t)c.word.size() - code_len);
    // Sort by savings descending
    std::sort(cands.begin(), cands.end(), [](auto &a, auto &b) {
      return a.savings > b.savings;
    });
    // Assign slots
    std::vector<Candidate> remaining;
    for (auto &c : cands) {
      if (slot >= tier_end || slot >= max_entries) {
        remaining.push_back(c);
        continue;
      }
      if (c.savings <= 0) {
        remaining.push_back(c);
        continue;
      }
      dict.word_to_idx[c.word] = slot;
      dict.idx_to_word.push_back(c.word);
      if (c.word.size() > dict.longest_word)
        dict.longest_word = (unsigned int)c.word.size();
      total_savings += c.savings;
      slot++;
    }
    cands = std::move(remaining);
  };

  fill_tier(kBoundary1, 1);
  fill_tier(kBoundary2, 2);
  fill_tier(kBoundary3, 3);

  if (verbose) {
    std::fprintf(stderr, "Dictionary: %d entries, estimated savings: %lld bytes\n",
                 dict.size(), (long long)total_savings);
    std::fprintf(stderr, "  Tier 1 (1-byte): %d entries\n", std::min(dict.size(), kBoundary1));
    std::fprintf(stderr, "  Tier 2 (2-byte): %d entries\n",
                 std::max(0, std::min(dict.size(), kBoundary2) - kBoundary1));
    std::fprintf(stderr, "  Tier 3 (3-byte): %d entries\n",
                 std::max(0, dict.size() - kBoundary2));
    std::fprintf(stderr, "Top 20 words:\n");
    for (int i = 0; i < std::min(20, dict.size()); ++i) {
      auto &w = dict.idx_to_word[i];
      std::fprintf(stderr, "  [%d] '%s' (code=%d bytes)\n",
                   i, w.c_str(), code_byte_length(i));
    }
  }

  return dict;
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------
static bool load_file(const std::string &path, std::string &data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.seekg(0, std::ios::end);
  size_t sz = in.tellg();
  in.seekg(0, std::ios::beg);
  data.resize(sz);
  in.read(&data[0], sz);
  return true;
}

static bool save_file(const std::string &path, const std::string &data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(data.data(), (std::streamsize)data.size());
  return (bool)out;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
static int cmd_build_dict(const std::string &input_path,
                          const std::string &dict_path,
                          int max_entries, bool verbose) {
  std::fprintf(stderr, "Loading corpus: %s\n", input_path.c_str());
  std::string data;
  if (!load_file(input_path, data)) {
    std::fprintf(stderr, "Cannot open %s\n", input_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "Corpus size: %zu bytes\n", data.size());

  std::fprintf(stderr, "Scanning word frequencies...\n");
  DictBuilder builder;
  builder.scan(data);
  auto candidates = builder.get_sorted();
  std::fprintf(stderr, "Unique words found: %zu\n", candidates.size());

  std::fprintf(stderr, "Building optimal dictionary (max %d entries)...\n", max_entries);
  DictCodec dict = build_optimal_dict(candidates, max_entries, verbose);

  if (!dict.save_cmix_format(dict_path)) {
    std::fprintf(stderr, "Failed to save dictionary to %s\n", dict_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "Saved dictionary: %s (%d entries)\n", dict_path.c_str(), dict.size());

  if (data.size() > 0) {
    HBEncoder enc(dict);
    auto stats = enc.compute_stats(data);
    std::fprintf(stderr, "\nQuick stats on training corpus:\n");
    std::fprintf(stderr, "  Words seen:     %lld\n", (long long)stats.total_words_seen);
    std::fprintf(stderr, "  Words encoded:  %lld (%.1f%%)\n",
                 (long long)stats.words_encoded,
                 100.0 * stats.words_encoded / std::max((int64_t)1, stats.total_words_seen));
    std::fprintf(stderr, "  Raw size:       %lld bytes\n", (long long)stats.bytes_raw);
    std::fprintf(stderr, "  Encoded size:   %lld bytes\n", (long long)stats.bytes_encoded);
    std::fprintf(stderr, "  Saved:          %lld bytes (%.2f%%)\n",
                 (long long)(stats.bytes_raw - stats.bytes_encoded),
                 100.0 * (stats.bytes_raw - stats.bytes_encoded) / std::max((int64_t)1, stats.bytes_raw));
  }

  return 0;
}

static int cmd_compress(const std::string &input_path,
                        const std::string &dict_path,
                        const std::string &output_path) {
  DictCodec dict;
  if (!dict.load_cmix_format(dict_path)) {
    std::fprintf(stderr, "Cannot load dictionary: %s\n", dict_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "Dictionary loaded: %d entries\n", dict.size());

  std::string data;
  if (!load_file(input_path, data)) {
    std::fprintf(stderr, "Cannot open input: %s\n", input_path.c_str());
    return 1;
  }

  HBEncoder enc(dict);
  std::string encoded = enc.encode(data);

  if (!save_file(output_path, encoded)) {
    std::fprintf(stderr, "Cannot write output: %s\n", output_path.c_str());
    return 1;
  }

  std::fprintf(stderr, "Input:   %zu bytes\n", data.size());
  std::fprintf(stderr, "Output:  %zu bytes (%.2f%% of original)\n",
               encoded.size(), 100.0 * encoded.size() / data.size());
  std::fprintf(stderr, "Entropy: %.4f bits/byte\n", byte_entropy(encoded));
  std::fprintf(stderr, "Alphabet: %d distinct bytes\n", count_distinct_bytes(encoded));
  return 0;
}

static int cmd_decompress(const std::string &input_path,
                          const std::string &dict_path,
                          const std::string &output_path) {
  DictCodec dict;
  if (!dict.load_cmix_format(dict_path)) {
    std::fprintf(stderr, "Cannot load dictionary: %s\n", dict_path.c_str());
    return 1;
  }

  std::string data;
  if (!load_file(input_path, data)) {
    std::fprintf(stderr, "Cannot open input: %s\n", input_path.c_str());
    return 1;
  }

  HBDecoder dec(dict);
  std::string decoded = dec.decode(data);

  if (!save_file(output_path, decoded)) {
    std::fprintf(stderr, "Cannot write output: %s\n", output_path.c_str());
    return 1;
  }

  std::fprintf(stderr, "Input:   %zu bytes\n", data.size());
  std::fprintf(stderr, "Output:  %zu bytes\n", decoded.size());
  return 0;
}

static int cmd_analyze(const std::string &input_path,
                       const std::string &dict_path) {
  DictCodec dict;
  if (!dict.load_cmix_format(dict_path)) {
    std::fprintf(stderr, "Cannot load dictionary: %s\n", dict_path.c_str());
    return 1;
  }

  std::string data;
  if (!load_file(input_path, data)) {
    std::fprintf(stderr, "Cannot open input: %s\n", input_path.c_str());
    return 1;
  }

  HBEncoder enc(dict);
  std::string encoded = enc.encode(data);
  auto stats = enc.compute_stats(data);

  HBDecoder dec(dict);
  std::string roundtrip = dec.decode(encoded);
  bool rt_ok = (roundtrip == data);

  std::printf("=== Dictionary Analysis ===\n");
  std::printf("Dictionary:       %s (%d entries)\n", dict_path.c_str(), dict.size());
  std::printf("Input:            %s (%zu bytes)\n", input_path.c_str(), data.size());
  std::printf("Roundtrip:        %s\n", rt_ok ? "OK" : "FAILED");
  std::printf("\n");
  std::printf("--- Raw input ---\n");
  std::printf("  Size:           %zu bytes\n", data.size());
  std::printf("  Entropy:        %.4f bits/byte\n", byte_entropy(data));
  std::printf("  Alphabet:       %d distinct bytes\n", count_distinct_bytes(data));
  std::printf("  Est. compress:  %.0f bytes\n", data.size() * byte_entropy(data) / 8.0);
  std::printf("\n");
  std::printf("--- Encoded output ---\n");
  std::printf("  Size:           %zu bytes (%.2f%% of original)\n",
              encoded.size(), 100.0 * encoded.size() / data.size());
  std::printf("  Entropy:        %.4f bits/byte\n", byte_entropy(encoded));
  std::printf("  Alphabet:       %d distinct bytes\n", count_distinct_bytes(encoded));
  std::printf("  Est. compress:  %.0f bytes\n", encoded.size() * byte_entropy(encoded) / 8.0);
  std::printf("\n");
  std::printf("--- Word statistics ---\n");
  std::printf("  Total words:    %lld\n", (long long)stats.total_words_seen);
  std::printf("  Encoded:        %lld (%.1f%%)\n", (long long)stats.words_encoded,
              100.0 * stats.words_encoded / std::max((int64_t)1, stats.total_words_seen));
  std::printf("  Bytes saved:    %lld\n", (long long)(stats.bytes_raw - stats.bytes_encoded));
  std::printf("\n");

  // Byte distribution of encoded output
  uint64_t freq[256] = {};
  for (unsigned char c : encoded) freq[c]++;
  std::vector<std::pair<int, uint64_t>> bfreq;
  for (int i = 0; i < 256; ++i) if (freq[i]) bfreq.push_back({i, freq[i]});
  std::sort(bfreq.begin(), bfreq.end(), [](auto &a, auto &b) { return a.second > b.second; });
  std::printf("--- Top 20 bytes in encoded output ---\n");
  for (int i = 0; i < std::min(20, (int)bfreq.size()); ++i) {
    int b = bfreq[i].first;
    uint64_t cnt = bfreq[i].second;
    double pct = 100.0 * cnt / encoded.size();
    char ch_buf[8];
    if (b >= 32 && b < 127) std::snprintf(ch_buf, sizeof(ch_buf), "'%c'", (char)b);
    else std::snprintf(ch_buf, sizeof(ch_buf), "0x%02X", b);
    std::printf("  byte %3d (%6s): %10llu (%5.2f%%)\n", b, ch_buf, (unsigned long long)cnt, pct);
  }

  uint64_t control = 0, ws = 0, printable = 0, high = 0;
  for (int i = 0; i < 256; ++i) {
    if (i < 32 && i != 9 && i != 10 && i != 13) control += freq[i];
    else if (i == 9 || i == 10 || i == 13 || i == 32) ws += freq[i];
    else if (i >= 33 && i <= 126) printable += freq[i];
    else if (i >= 128) high += freq[i];
  }
  std::printf("\n--- Byte class distribution ---\n");
  std::printf("  Control (no ws): %10llu (%5.2f%%)\n", (unsigned long long)control, 100.0*control/encoded.size());
  std::printf("  Whitespace:      %10llu (%5.2f%%)\n", (unsigned long long)ws, 100.0*ws/encoded.size());
  std::printf("  Printable ASCII: %10llu (%5.2f%%)\n", (unsigned long long)printable, 100.0*printable/encoded.size());
  std::printf("  High (>=0x80):   %10llu (%5.2f%%)\n", (unsigned long long)high, 100.0*high/encoded.size());

  return 0;
}

static int cmd_verify(const std::string &input_path,
                      const std::string &dict_path) {
  DictCodec dict;
  if (!dict.load_cmix_format(dict_path)) {
    std::fprintf(stderr, "Cannot load dictionary: %s\n", dict_path.c_str());
    return 1;
  }

  std::string data;
  if (!load_file(input_path, data)) {
    std::fprintf(stderr, "Cannot open input: %s\n", input_path.c_str());
    return 1;
  }

  HBEncoder enc(dict);
  std::string encoded = enc.encode(data);

  HBDecoder dec(dict);
  std::string decoded = dec.decode(encoded);

  if (decoded == data) {
    std::printf("Roundtrip OK (%zu bytes -> %zu bytes -> %zu bytes)\n",
                data.size(), encoded.size(), decoded.size());
    return 0;
  } else {
    std::printf("Roundtrip FAILED!\n");
    std::printf("  Original: %zu bytes\n", data.size());
    std::printf("  Encoded:  %zu bytes\n", encoded.size());
    std::printf("  Decoded:  %zu bytes\n", decoded.size());
    size_t minlen = std::min(data.size(), decoded.size());
    for (size_t i = 0; i < minlen; ++i) {
      if (data[i] != decoded[i]) {
        std::printf("  First diff at byte %zu: orig=0x%02X decoded=0x%02X\n",
                    i, (unsigned char)data[i], (unsigned char)decoded[i]);
        size_t ctx_start = i > 30 ? i - 30 : 0;
        size_t ctx_end = std::min(data.size(), i + 30);
        std::printf("  orig context: ");
        for (size_t j = ctx_start; j < ctx_end; ++j) {
          unsigned char c = data[j];
          if (c >= 32 && c < 127) std::printf("%c", c);
          else std::printf("\\x%02X", c);
        }
        std::printf("\n  dec  context: ");
        for (size_t j = ctx_start; j < std::min(decoded.size(), ctx_end); ++j) {
          unsigned char c = decoded[j];
          if (c >= 32 && c < 127) std::printf("%c", c);
          else std::printf("\\x%02X", c);
        }
        std::printf("\n");
        break;
      }
    }
    return 1;
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void print_usage() {
  std::printf(
    "dictionary_compress -- High-byte dictionary preprocessor\n"
    "\n"
    "Usage:\n"
    "  build-dict  --input CORPUS --dict OUTPUT.dic [--max-entries N] [--verbose]\n"
    "  compress    --input FILE --dict DICT.dic --output OUT.bin\n"
    "  decompress  --input FILE --dict DICT.dic --output OUT.txt\n"
    "  analyze     --input FILE --dict DICT.dic\n"
    "  verify      --input FILE --dict DICT.dic\n"
    "\n"
    "The dictionary format is one-word-per-line, compatible with cmix english.dic.\n"
    "Words are ordered by dictionary index (position = code assignment).\n"
    "The first 80 words get 1-byte codes (most valuable slots).\n"
  );
}

int main(int argc, char **argv) {
  if (argc < 2) { print_usage(); return 0; }

  std::string cmd = argv[1];
  std::string input, dict_path, output;
  int max_entries = kMaxDictSize;
  bool verbose = false;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", argv[i]); std::exit(1); }
      return std::string(argv[++i]);
    };
    if (a == "--input") input = next();
    else if (a == "--dict") dict_path = next();
    else if (a == "--output") output = next();
    else if (a == "--max-entries") max_entries = std::atoi(next().c_str());
    else if (a == "--verbose") verbose = true;
    else std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
  }

  if (cmd == "build-dict") {
    if (input.empty() || dict_path.empty()) { print_usage(); return 1; }
    return cmd_build_dict(input, dict_path, max_entries, verbose);
  } else if (cmd == "compress") {
    if (input.empty() || dict_path.empty() || output.empty()) { print_usage(); return 1; }
    return cmd_compress(input, dict_path, output);
  } else if (cmd == "decompress") {
    if (input.empty() || dict_path.empty() || output.empty()) { print_usage(); return 1; }
    return cmd_decompress(input, dict_path, output);
  } else if (cmd == "analyze") {
    if (input.empty() || dict_path.empty()) { print_usage(); return 1; }
    return cmd_analyze(input, dict_path);
  } else if (cmd == "verify") {
    if (input.empty() || dict_path.empty()) { print_usage(); return 1; }
    return cmd_verify(input, dict_path);
  } else {
    print_usage();
    return 1;
  }
}
