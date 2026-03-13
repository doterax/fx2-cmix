#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Prototype dictionary-based compressor with optional embedding-inspired
// preprocessor Single-file, modular classes, UTF-8-friendly tokenization (ASCII
// word set), reversible compression with case markers and escaping.
//
// Markers (escape alphabet)
#define MC_LOWER   '~'  // lowercase dictionary word
#define MC_CAP1    '^'  // capitalize first letter
#define MC_ALLCAPS '*'  // all letters uppercase
#define MC_ESCAPE  '\\' // escape prefix for raw data blocks ("\\<len>:\"...\"")
// Note: We use a length-prefixed raw segment to avoid ambiguity with marker
// stream.
//       This simplifies robust decompression.

static inline bool is_word_char(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') /*|| (c == '\'')*/;
}

static inline bool is_ascii_letter(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Base62 encoding (a-zA-Z0-9) as requested; 'a' is first code in examples.
// We'll use ordering: a..z, A..Z, 0..9
static const char BASE62_ALPHABET[] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static inline std::string base62_encode(uint64_t v) {
  if (v == 0)
    return std::string(1, BASE62_ALPHABET[0]);
  char buf[32];
  int  pos = 31;
  buf[pos] = '\0';
  --pos;
  while (v > 0 && pos >= 0) {
    uint64_t r = v % 62ULL;
    v /= 62ULL;
    buf[pos--] = BASE62_ALPHABET[r];
  }
  return std::string(&buf[pos + 1]);
}
static inline int base62_index(char c) {
  if (c >= 'a' && c <= 'z')
    return c - 'a';
  if (c >= 'A' && c <= 'Z')
    return 26 + (c - 'A');
  if (c >= '0' && c <= '9')
    return 52 + (c - '0');
  return -1;
}
static inline bool is_base62(char c) {
  return base62_index(c) >= 0;
}
static inline bool base62_decode(const std::string &s, uint64_t &out) {
  uint64_t v = 0;
  for (char c : s) {
    int idx = base62_index(c);
    if (idx < 0)
      return false;
    v = v * 62ULL + (uint64_t)idx;
  }
  out = v;
  return true;
}

// Case pattern detection for tokens containing ASCII letters
enum class CasePattern { Lower, Capitalized, Upper, Mixed, None };
static CasePattern detect_case(const std::string &token) {
  bool any = false, any_lower = false, any_upper = false;
  for (unsigned char c : token)
    if (is_ascii_letter(c)) {
      any = true;
      if (c >= 'A' && c <= 'Z')
        any_upper = true;
      else
        any_lower = true;
    }
  if (!any)
    return CasePattern::None;
  // Check capitalized: first alpha upper, rest alpha lower
  int first_alpha = -1;
  for (size_t i = 0; i < token.size(); ++i) {
    unsigned char c = token[i];
    if (is_ascii_letter(c)) {
      first_alpha = (int)i;
      break;
    }
  }
  if (first_alpha >= 0) {
    bool cap1       = (token[first_alpha] >= 'A' && token[first_alpha] <= 'Z');
    bool rest_lower = true;
    for (size_t i = first_alpha + 1; i < token.size(); ++i) {
      unsigned char c = token[i];
      if (is_ascii_letter(c) && !(c >= 'a' && c <= 'z')) {
        rest_lower = false;
        break;
      }
    }
    if (cap1 && rest_lower)
      return CasePattern::Capitalized;
  }
  if (any_upper && !any_lower)
    return CasePattern::Upper;
  if (any_lower && !any_upper)
    return CasePattern::Lower;
  return CasePattern::Mixed;
}
static inline std::string to_lower_ascii(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s)
    out.push_back((char)std::tolower(c));
  return out;
}
static inline std::string apply_case_pattern(const std::string &lower_word,
                                             CasePattern        pat) {
  if (pat == CasePattern::Lower || pat == CasePattern::None)
    return lower_word;
  if (pat == CasePattern::Upper) {
    std::string w = lower_word;
    for (auto &ch : w)
      if (is_ascii_letter((unsigned char)ch))
        ch = (char)std::toupper((unsigned char)ch);
    return w;
  }
  if (pat == CasePattern::Capitalized) {
    std::string w = lower_word;
    for (size_t i = 0; i < w.size(); ++i) {
      unsigned char c = w[i];
      if (is_ascii_letter(c)) {
        w[i] = (char)std::toupper(c);
        break;
      }
    }
    return w;
  }
  // Mixed: we don't reconstruct mixed with markers; compressor emits raw
  return lower_word;
}

// Tokenizer: splits into word tokens and non-word spans; also counts articles
// via "<page>"
class Tokenizer {
public:
  struct Span {
    bool        is_word;
    std::string text;
    size_t      start_pos;
    size_t      article_index;
  };
  static std::vector<Span> Tokenize(const std::string &data,
                                    size_t            &article_count) {
    std::vector<Span> spans;
    spans.reserve(data.size() / 8);
    article_count      = 0;
    size_t cur_article = 0;
    // simple article counting
    for (size_t i = 0; i + 5 <= data.size(); ++i)
      if (data.compare(i, 6, "<page>") == 0)
        ++article_count;
    // assign running article_index
    cur_article             = 0;
    size_t i                = 0;
    size_t last_article_pos = 0;
    while (i < data.size()) {
      if (i + 5 <= data.size() && data.compare(i, 6, "<page>") == 0) {
        spans.push_back({false, std::string("<page>"), i, cur_article});
        ++cur_article;
        i += 6;
        continue;
      }
      size_t start = i;
      if (is_word_char((unsigned char)data[i])) {
        while (i < data.size() && is_word_char((unsigned char)data[i]))
          ++i;
        spans.push_back(
            {true, data.substr(start, i - start), start, cur_article});
      } else {
        while (i < data.size() && !is_word_char((unsigned char)data[i])) {
          if (i + 5 <= data.size() && data.compare(i, 6, "<page>") == 0)
            break;
          ++i;
        }
        spans.push_back(
            {false, data.substr(start, i - start), start, cur_article});
      }
    }
    if (article_count == 0 && data.size() > 0)
      article_count = 1; // at least 1
    return spans;
  }
};

// Dictionary building
struct DictEntry {
  std::string           word;
  uint64_t              count = 0;
  std::vector<uint32_t> positions;
  std::vector<uint32_t> articles;
};

class DictionaryBuilder {
public:
  struct Options {
    bool     lowercase     = true;
    uint64_t min_frequency = 1;
    uint32_t position_cap  = 0;
  };
  DictionaryBuilder(const Options &opt) : opt_(opt) {}
  void ingest_spans(const std::vector<Tokenizer::Span> &spans) {
    uint32_t token_index = 0;
    for (const auto &sp : spans) {
      if (!sp.is_word)
        continue;
      std::string w = opt_.lowercase ? to_lower_ascii(sp.text) : sp.text;
      auto       &e = map_[w];
      e.word        = w;
      e.count++;
      if (opt_.position_cap == 0 || e.positions.size() < opt_.position_cap)
        e.positions.push_back(token_index);
      if (e.articles.empty() || e.articles.back() != sp.article_index)
        e.articles.push_back((uint32_t)sp.article_index);
      ++token_index;
    }
  }
  void finalize(std::vector<DictEntry> &out_entries) {
    out_entries.clear();
    out_entries.reserve(map_.size());
    for (auto &kv : map_)
      if (kv.second.count >= opt_.min_frequency)
        out_entries.push_back(std::move(kv.second));
    std::sort(out_entries.begin(), out_entries.end(),
              [](const DictEntry &a, const DictEntry &b) {
                if (a.count != b.count)
                  return a.count > b.count;
                return a.word < b.word;
              });
  }

  // Augment with root+suffix entries derived from words (heuristic)
  static void augment_with_suffixes(std::vector<DictEntry> &entries,
                                    uint32_t                min_root_len   = 4,
                                    uint32_t                min_suffix_len = 2,
                                    uint32_t                max_suffix_len = 6,
                                    uint64_t                min_root_total = 3,
                                    uint32_t                min_variants = 2) {
    // Build frequency map of words
    std::unordered_map<std::string, uint64_t> word_freq;
    word_freq.reserve(entries.size() * 2);
    for (auto &e : entries)
      word_freq[e.word] = e.count;

    struct RootAgg {
      uint64_t                                  total = 0;
      std::unordered_map<std::string, uint64_t> suffixes;
    };
    std::unordered_map<std::string, RootAgg> roots;
    roots.reserve(entries.size() * 2);
    for (auto &e : entries) {
      const std::string &w = e.word;
      if (w.size() < min_root_len + min_suffix_len)
        continue;
      for (uint32_t slen = min_suffix_len; slen <= max_suffix_len; ++slen) {
        if (slen >= w.size())
          break;
        std::string root = w.substr(0, w.size() - slen);
        if (root.size() < min_root_len)
          continue;
        std::string suf = w.substr(w.size() - slen);
        auto       &agg = roots[root];
        agg.total += e.count;
        agg.suffixes[suf] += e.count;
      }
    }
    // Select productive roots
    std::vector<std::pair<std::string, RootAgg>> selected;
    selected.reserve(roots.size());
    for (auto &kv : roots) {
      if (kv.second.total >= min_root_total &&
          kv.second.suffixes.size() >= min_variants)
        selected.push_back(kv);
    }
    if (selected.empty())
      return;

    // Add root and suffix entries to dictionary entries if not present; update
    // counts if present
    std::unordered_map<std::string, size_t> index;
    index.reserve(entries.size() * 2);
    for (size_t i = 0; i < entries.size(); ++i)
      index[entries[i].word] = i;
    auto ensure_entry = [&](const std::string &s, uint64_t add_count) {
      auto it = index.find(s);
      if (it == index.end()) {
        DictEntry e;
        e.word  = s;
        e.count = add_count;
        entries.push_back(std::move(e));
        index[s] = entries.size() - 1;
      } else {
        entries[it->second].count += add_count;
      }
    };
    for (auto &kv : selected) {
      const std::string &root = kv.first;
      const RootAgg     &ra   = kv.second;
      ensure_entry(root, ra.total);
      for (auto &sv : ra.suffixes)
        ensure_entry(sv.first, sv.second);
    }
    // Re-sort by frequency desc then word
    std::sort(entries.begin(), entries.end(),
              [](const DictEntry &a, const DictEntry &b) {
                if (a.count != b.count)
                  return a.count > b.count;
                return a.word < b.word;
              });
  }

private:
  Options                                    opt_;
  std::unordered_map<std::string, DictEntry> map_;
};

// Co-occurrence graph and optional embedding-like reorder
class EmbeddingPreprocessor {
public:
  struct Options {
    int   window     = 4;
    bool  enable     = false;
    int   dims       = 64;
    int   iterations = 128;
    float move_coeff = 0.1f;
  };
  EmbeddingPreprocessor(const Options &opt) : opt_(opt) {}

  void build_graph(const std::vector<Tokenizer::Span> &spans) {
    // Build token sequence of lowercased words
    sequence_.clear();
    sequence_.reserve(spans.size());
    for (const auto &sp : spans)
      if (sp.is_word)
        sequence_.push_back(to_lower_ascii(sp.text));
    // map ids
    for (const auto &w : sequence_) {
      auto it = id_.find(w);
      if (it == id_.end()) {
        int nid = (int)id_.size();
        id_[w]  = nid;
        names_.push_back(w);
      }
    }
    int V = (int)names_.size();
    adj_.assign(V, {});
    // Sliding window co-occurrence counts
    for (size_t i = 0; i < sequence_.size(); ++i) {
      int wi = id_[sequence_[i]];
      int lo = (int)(i > (size_t)opt_.window ? i - opt_.window : 0);
      int hi = (int)std::min(sequence_.size(), i + (size_t)opt_.window + 1);
      for (int j = lo; j < hi; ++j)
        if ((size_t)j != i) {
          int wj = id_[sequence_[j]];
          adj_[wi][wj] += 1.0f;
        }
    }
  }

  // Greedy neighbor ordering based on co-occurrence strength
  std::vector<int>
  greedy_order_by_graph(const std::vector<DictEntry> &dict) const {
    // seed ordering by dict frequency list: map name->rank
    std::unordered_map<std::string, int> freq_rank;
    freq_rank.reserve(dict.size());
    for (size_t i = 0; i < dict.size(); ++i)
      freq_rank[dict[i].word] = (int)i;
    int               V = (int)names_.size();
    std::vector<char> used(V, 0);
    std::vector<int>  order;
    order.reserve(V);
    // choose start as highest freq present in names_
    int start     = -1;
    int best_rank = INT_MAX;
    for (int vid = 0; vid < V; ++vid) {
      auto it = freq_rank.find(names_[vid]);
      if (it == freq_rank.end())
        continue;
      if (it->second < best_rank) {
        best_rank = it->second;
        start     = vid;
      }
    }
    if (start < 0) { // fallback id order
      for (int v = 0; v < V; ++v)
        order.push_back(v);
      return order;
    }
    int cur   = start;
    used[cur] = 1;
    order.push_back(cur);
    for (;;) {
      // pick strongest neighbor not used
      int   next  = -1;
      float bestw = 0.0f;
      for (auto &kv : adj_[cur]) {
        int v = kv.first;
        if (used[v])
          continue;
        float w = kv.second;
        if (w > bestw) {
          bestw = w;
          next  = v;
        }
      }
      if (next < 0) {
        // find next most frequent unused
        int best  = -1;
        int bestR = INT_MAX;
        for (int v = 0; v < V; ++v)
          if (!used[v]) {
            auto it = freq_rank.find(names_[v]);
            int  r  = it == freq_rank.end() ? INT_MAX : it->second;
            if (r < bestR) {
              bestR = r;
              best  = v;
            }
          }
        if (best < 0)
          break;
        next = best;
      }
      used[next] = 1;
      order.push_back(next);
      cur = next;
    }
    return order;
  }

  // Optional lightweight embedding relaxation: pull vectors toward neighbor
  // means
  std::vector<int> embed_and_order(const std::vector<DictEntry> &dict) const {
    int V = (int)names_.size();
    if (V == 0)
      return {};
    // Lightweight embedding without external deps: random init + neighbor mean
    // relax
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<std::vector<float>>       pos(
        V, std::vector<float>(std::max(1, opt_.dims), 0.f));
    for (int i = 0; i < V; ++i)
      for (int d = 0; d < opt_.dims; ++d)
        pos[i][d] = dist(rng);
    for (int it = 0; it < opt_.iterations; ++it) {
      for (int i = 0; i < V; ++i) {
        if (adj_[i].empty())
          continue;
        std::vector<float> mean(opt_.dims, 0.f);
        float              wsum = 0.f;
        for (auto &kv : adj_[i]) {
          int   j = kv.first;
          float w = kv.second;
          wsum += w;
          for (int d = 0; d < opt_.dims; ++d)
            mean[d] += pos[j][d] * w;
        }
        if (wsum > 0) {
          float mc = opt_.move_coeff;
          for (int d = 0; d < opt_.dims; ++d) {
            mean[d] /= wsum;
            pos[i][d] = pos[i][d] + mc * (mean[d] - pos[i][d]);
          }
        }
      }
    }
    struct Item {
      int   id;
      float score;
    };
    std::vector<Item> arr;
    arr.reserve(V);
    for (int i = 0; i < V; ++i) {
      float s = 0.f;
      for (int d = 0; d < opt_.dims; ++d)
        s += pos[i][d];
      arr.push_back({i, s});
    }
    std::sort(arr.begin(), arr.end(),
              [](const Item &a, const Item &b) { return a.score > b.score; });
    std::vector<int> order;
    order.reserve(V);
    for (auto &it : arr)
      order.push_back(it.id);
    return order;
  }
  // Produce a reordered list of dictionary words according to
  // embedding/co-occurrence
  std::vector<std::string>
  reorder_words(const std::vector<DictEntry> &dict) const {
    std::unordered_set<std::string> dictset;
    for (auto &e : dict)
      dictset.insert(e.word);
    std::vector<int> ord = greedy_order_by_graph(dict);
    if (opt_.enable) {
      std::vector<int> emb = embed_and_order(dict);
      if (!emb.empty())
        ord = emb; // prefer embedding order when available
    }
    std::vector<std::string> words;
    words.reserve(dict.size());
    std::unordered_set<std::string> seen;
    for (int id : ord) {
      const std::string &w = names_[id];
      if (dictset.count(w) && !seen.count(w)) {
        words.push_back(w);
        seen.insert(w);
      }
    }
    for (auto &e : dict)
      if (!seen.count(e.word))
        words.push_back(e.word);
    return words;
  }

private:
  Options                                     opt_;
  std::vector<std::string>                    sequence_;
  std::unordered_map<std::string, int>        id_;
  std::vector<std::string>                    names_;
  std::vector<std::unordered_map<int, float>> adj_;
};

// Dictionary codec
struct Dictionary {
  // maps
  std::unordered_map<std::string, uint64_t> word_to_id; // lowercased word -> id
  std::unordered_map<uint64_t, std::string> id_to_word;

  static bool Load(const std::string &path, Dictionary &d) {
    d.word_to_id.clear();
    d.id_to_word.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in)
      return false;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      // format: word,code
      size_t comma = line.rfind(',');
      if (comma == std::string::npos)
        continue;
      std::string word = line.substr(0, comma);
      std::string code = line.substr(comma + 1);
      uint64_t    id   = 0;
      if (!base62_decode(code, id))
        continue;
      d.word_to_id[word] = id;
      d.id_to_word[id]   = word;
    }
    return true;
  }

  static bool
  Save(const std::string                                   &path,
       const std::vector<std::pair<std::string, uint64_t>> &entries) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
      return false;
    for (auto &kv : entries) {
      out << kv.first << "," << base62_encode(kv.second) << "\n";
    }
    return true;
  }
};

// Compressor/Decompressor
class Compressor {
public:
  Compressor(const Dictionary &dict, bool enable_suffix = false,
             bool prefer_suffix = false, bool force_suffix = false)
      : dict_(dict), enable_suffix_(enable_suffix),
        prefer_suffix_(prefer_suffix), force_suffix_(force_suffix) {}

  // Compress to text stream with markers. Non-word spans written as-is, but any
  // marker characters in raw are escaped using a length-prefixed raw segment
  // format: "\\<len>:<raw>" when needed.
  std::string compress(const std::vector<Tokenizer::Span> &spans) const {
    std::string out;
    out.reserve(spans.size() * 4);
    for (const auto &sp : spans) {
      if (!sp.is_word) {
        append_raw(out, sp.text);
        continue;
      }
      // try dictionary
      CasePattern pat = detect_case(sp.text);
      if (pat == CasePattern::Mixed) {
        append_raw(out, sp.text);
        continue;
      }
      std::string lower = to_lower_ascii(sp.text);
      // choose best encoding: whole-word vs root+suffix (if enabled)
      auto encode_len = [](char marker, uint64_t id) {
        return (size_t)1 + base62_encode(id).size();
      };
      auto     it          = dict_.word_to_id.find(lower);
      size_t   best_len    = SIZE_MAX;
      int      choice      = 0; // 0=raw,1=whole,2=split
      bool     split_found = false;
      uint64_t id_w = 0, id_root = 0, id_suf = 0;
      size_t   split_pos = 0;

      // whole-word candidate
      if (it != dict_.word_to_id.end()) {
        id_w     = it->second;
        size_t l = encode_len(MC_LOWER, id_w);
        best_len = l;
        choice   = 1;
      }
      // split candidate
      if (enable_suffix_ && lower.size() >= 6) {
        for (size_t slen = 2; slen <= 6 && slen < lower.size(); ++slen) {
          std::string root = lower.substr(0, lower.size() - slen);
          std::string suf  = lower.substr(lower.size() - slen);
          auto        ir   = dict_.word_to_id.find(root);
          if (ir == dict_.word_to_id.end())
            continue;
          auto is = dict_.word_to_id.find(suf);
          if (is == dict_.word_to_id.end())
            continue;
          size_t l;
          if (pat == CasePattern::Upper) {
            // mark both to ensure full caps
            l = encode_len(MC_ALLCAPS, ir->second) +
                encode_len(MC_ALLCAPS, is->second);
          } else if (pat == CasePattern::Capitalized) {
            // Capitalize the first alphabetic letter across root+suffix.
            size_t root_len      = lower.size() - slen;
            bool   alpha_in_root = false;
            for (size_t k = 0; k < root_len; ++k) {
              if (is_ascii_letter((unsigned char)lower[k])) {
                alpha_in_root = true;
                break;
              }
            }
            char m1 = alpha_in_root ? MC_CAP1 : MC_LOWER;
            char m2 = alpha_in_root ? MC_LOWER : MC_CAP1;
            l       = encode_len(m1, ir->second) + encode_len(m2, is->second);
          } else {
            // lower case: both pieces lower
            l = encode_len(MC_LOWER, ir->second) +
                encode_len(MC_LOWER, is->second);
          }
          // prefer or force suffix splits if requested
          if (force_suffix_ || l < best_len ||
              (prefer_suffix_ && choice == 1)) {
            best_len    = l;
            choice      = 2;
            id_root     = ir->second;
            id_suf      = is->second;
            split_pos   = slen;
            split_found = true;
          }
        }
      }
      // Savings rule: only encode if shorter than raw by at least 1
      size_t raw_len = sp.text.size();
      if (choice == 0 || best_len >= raw_len) {
        append_raw(out, sp.text);
        continue;
      }
      if (choice == 1 || (force_suffix_ && !split_found)) {
        char marker = MC_LOWER;
        if (pat == CasePattern::Capitalized)
          marker = MC_CAP1;
        else if (pat == CasePattern::Upper)
          marker = MC_ALLCAPS;
        out.push_back(marker);
        out += base62_encode(id_w);
      } else {
        if (pat == CasePattern::Upper) {
          out.push_back(MC_ALLCAPS);
          out += base62_encode(id_root);
          out.push_back(MC_ALLCAPS);
          out += base62_encode(id_suf);
        } else if (pat == CasePattern::Capitalized) {
          // Ensure capitalization applies to the first alphabetic letter
          size_t root_len      = lower.size() - split_pos;
          bool   alpha_in_root = false;
          for (size_t k = 0; k < root_len; ++k) {
            if (is_ascii_letter((unsigned char)lower[k])) {
              alpha_in_root = true;
              break;
            }
          }
          char m1 = alpha_in_root ? MC_CAP1 : MC_LOWER;
          char m2 = alpha_in_root ? MC_LOWER : MC_CAP1;
          out.push_back(m1);
          out += base62_encode(id_root);
          out.push_back(m2);
          out += base62_encode(id_suf);
        } else {
          out.push_back(MC_LOWER);
          out += base62_encode(id_root);
          out.push_back(MC_LOWER);
          out += base62_encode(id_suf);
        }
      }
    }
    return out;
  }

private:
  const Dictionary  &dict_;
  bool               enable_suffix_ = false;
  bool               prefer_suffix_ = false;
  bool               force_suffix_  = false;

  static inline bool needs_escape(const std::string &s) {
    for (unsigned char c : s)
      if (c == MC_LOWER || c == MC_CAP1 || c == MC_ALLCAPS || c == MC_ESCAPE)
        return true;
    return false;
  }
  static void append_raw(std::string &out, const std::string &raw) {
    if (!needs_escape(raw)) {
      out += raw;
      return;
    }
    // length-prefixed: \\<len>:<raw>
    out.push_back(MC_ESCAPE);
    out += std::to_string((unsigned long long)raw.size());
    out.push_back(':');
    out += raw;
  }
};

class Decompressor {
public:
  Decompressor(const Dictionary &dict) : dict_(dict) {}

  std::string decompress(const std::string &data) const {
    std::string out;
    out.reserve(data.size());
    size_t i = 0;
    while (i < data.size()) {
      char c = data[i];
      if (c == MC_ESCAPE) {
        // read number until ':'
        size_t j = i + 1;
        size_t n = 0;
        while (j < data.size() && std::isdigit((unsigned char)data[j])) {
          n = n * 10 + (data[j] - '0');
          ++j;
        }
        if (j < data.size() && data[j] == ':') {
          ++j;
          if (j + n <= data.size()) {
            out.append(data, j, n);
            i = j + n;
            continue;
          }
        }
        // malformed; emit as-is
        out.push_back(c);
        ++i;
        continue;
      } else if (c == MC_LOWER || c == MC_CAP1 || c == MC_ALLCAPS) {
        // parse base62 code
        size_t j = i + 1;
        while (j < data.size() && is_base62(data[j]))
          ++j;
        if (j == i + 1) {
          out.push_back(c);
          ++i;
          continue;
        }
        std::string code = data.substr(i + 1, j - (i + 1));
        uint64_t    id   = 0;
        if (!base62_decode(code, id)) {
          out.push_back(c);
          ++i;
          continue;
        }
        auto it = dict_.id_to_word.find(id);
        if (it == dict_.id_to_word.end()) {
          out.push_back(c);
          ++i;
          continue;
        }
        std::string w = it->second;
        if (c == MC_CAP1)
          w = apply_case_pattern(w, CasePattern::Capitalized);
        else if (c == MC_ALLCAPS)
          w = apply_case_pattern(w, CasePattern::Upper);
        out += w;
        i = j;
        continue;
      } else {
        out.push_back(c);
        ++i;
        continue;
      }
    }
    return out;
  }

private:
  const Dictionary &dict_;
};

// Pipeline runner for build-dict
static int cmd_build_dict(const std::string &input,
                          const std::string &dict_path, bool embed, bool suffix,
                          uint64_t min_freq, int window) {
  // Read input
  std::ifstream in(input, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "Cannot open %s\n", input.c_str());
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string                data          = ss.str();
  size_t                     article_count = 0;
  auto                       spans = Tokenizer::Tokenize(data, article_count);

  DictionaryBuilder::Options dbopt;
  dbopt.lowercase     = true;
  dbopt.min_frequency = min_freq;
  dbopt.position_cap  = 0;
  DictionaryBuilder db(dbopt);
  db.ingest_spans(spans);
  std::vector<DictEntry> entries;
  db.finalize(entries);
  if (suffix)
    DictionaryBuilder::augment_with_suffixes(entries);

  // Prune non-optimal entries and keep only useful roots/suffixes
  auto provisional_order = entries; // frequency-sorted
  std::unordered_map<std::string, uint64_t> provisional_id;
  for (size_t i = 0; i < provisional_order.size(); ++i)
    provisional_id[provisional_order[i].word] = (uint64_t)i;
  auto code_len = [&](const std::string &w) {
    auto it = provisional_id.find(w);
    if (it == provisional_id.end())
      return (size_t)SIZE_MAX;
    return (size_t)1 + base62_encode(it->second).size();
  };
  std::unordered_set<std::string> keep_words;
  std::unordered_set<std::string> keep_parts; // roots/suffixes contributing to useful splits
  // Mark original "whole words" that are actually token words (positions non-empty)
  for (auto &e : entries) {
    const std::string &w = e.word;
    size_t raw_len       = w.size();
    size_t enc_len       = code_len(w);
    bool   is_token_word = !e.positions.empty();
    if (is_token_word && enc_len < raw_len)
      keep_words.insert(w);
  }
  // Evaluate splits to mark useful roots/suffixes
  if (suffix) {
    std::unordered_set<std::string> entry_set;
    entry_set.reserve(entries.size() * 2);
    for (auto &e : entries)
      entry_set.insert(e.word);
    for (auto &e : entries) {
      if (e.positions.empty())
        continue; // only consider real token words as split targets
      const std::string &w = e.word;
      size_t             raw_len = w.size();
      if (w.size() >= 6) {
        for (size_t slen = 2; slen <= 6 && slen < w.size(); ++slen) {
          std::string root = w.substr(0, w.size() - slen);
          std::string suf  = w.substr(w.size() - slen);
          if (!entry_set.count(root) || !entry_set.count(suf))
            continue;
          size_t len_split = code_len(root) + code_len(suf);
          if (len_split < raw_len) {
            keep_parts.insert(root);
            keep_parts.insert(suf);
          }
        }
      }
    }
  }
  // Build pruned list
  std::vector<DictEntry> pruned;
  pruned.reserve(entries.size());
  for (auto &e : entries) {
    const std::string &w = e.word;
    bool is_token_word    = !e.positions.empty();
    if (is_token_word) {
      if (keep_words.count(w))
        pruned.push_back(e);
    } else { // augmented root/suffix entries
      if (keep_parts.count(w))
        pruned.push_back(e);
    }
  }
  entries.swap(pruned);

  // Assign IDs strictly by frequency to guarantee shortest codes
  // for the most common words (e.g., "the" gets id 0 => code 'a').
  std::vector<DictEntry> freq_sorted = entries;
  std::sort(freq_sorted.begin(), freq_sorted.end(), [](const DictEntry &a, const DictEntry &b) {
    if (a.count != b.count) return a.count > b.count;
    return a.word < b.word;
  });
  std::unordered_map<std::string, uint64_t> idmap;
  idmap.reserve(freq_sorted.size());
  for (size_t i = 0; i < freq_sorted.size(); ++i)
    idmap[freq_sorted[i].word] = (uint64_t)i;

  // Prepare output entries; sort alphabetically by word as requested
  std::vector<std::pair<std::string, uint64_t>> out;
  out.reserve(entries.size());
  for (auto &e : entries) {
    auto it = idmap.find(e.word);
    if (it != idmap.end())
      out.emplace_back(e.word, it->second);
  }
  std::sort(out.begin(), out.end(),
            [](auto &a, auto &b) { return a.first < b.first; });

  if (!Dictionary::Save(dict_path, out)) {
    std::fprintf(stderr, "Failed to save dict %s\n", dict_path.c_str());
    return 1;
  }
  std::printf("Dictionary entries: %zu\n", out.size());
  std::printf("Articles detected: %zu\n", article_count);
  return 0;
}

static bool load_file(const std::string &path, std::string &data) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  data = ss.str();
  return true;
}
static bool save_file(const std::string &path, const std::string &data) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  out.write(data.data(), (std::streamsize)data.size());
  return (bool)out;
}

static int cmd_compress(const std::string &input, const std::string &dict_path,
                        const std::string &output, bool suffix,
                        bool prefer_suffix, bool force_suffix) {
  std::string data;
  if (!load_file(input, data)) {
    std::fprintf(stderr, "Cannot open %s\n", input.c_str());
    return 1;
  }
  size_t     dummy = 0;
  auto       spans = Tokenizer::Tokenize(data, dummy);
  Dictionary dict;
  if (!Dictionary::Load(dict_path, dict)) {
    std::fprintf(stderr, "Cannot read dict %s\n", dict_path.c_str());
    return 1;
  }
  Compressor  comp(dict, suffix, prefer_suffix, force_suffix);
  std::string encoded = comp.compress(spans);
  if (!output.empty()) {
    if (!save_file(output, encoded)) {
      std::fprintf(stderr, "Cannot write %s\n", output.c_str());
      return 1;
    }
  } else {
    std::cout << encoded << std::endl;
  }
  return 0;
}

static int cmd_decompress(const std::string &input,
                          const std::string &dict_path,
                          const std::string &output) {
  std::string data;
  if (!load_file(input, data)) {
    std::fprintf(stderr, "Cannot open %s\n", input.c_str());
    return 1;
  }
  Dictionary dict;
  if (!Dictionary::Load(dict_path, dict)) {
    std::fprintf(stderr, "Cannot read dict %s\n", dict_path.c_str());
    return 1;
  }
  Decompressor dec(dict);
  std::string  plain = dec.decompress(data);
  if (!output.empty()) {
    if (!save_file(output, plain)) {
      std::fprintf(stderr, "Cannot write %s\n", output.c_str());
      return 1;
    }
  } else {
    std::cout << plain << std::endl;
  }
  return 0;
}

static int cmd_test(const std::string &str, const std::string &input,
                    const std::string &dict_out, bool embed, bool suffix,
                    bool prefer_suffix, bool force_suffix, uint64_t min_freq,
                    int window, const std::string &outbin) {
  std::string data;
  if (!str.empty())
    data = str;
  else if (!input.empty()) {
    if (!load_file(input, data)) {
      std::fprintf(stderr, "Cannot open %s\n", input.c_str());
      return 1;
    }
  } else {
    std::fprintf(stderr, "Provide --string or --input\n");
    return 1;
  }

  // Build dict in-memory and optionally save
  size_t                     article_count = 0;
  auto                       spans = Tokenizer::Tokenize(data, article_count);
  DictionaryBuilder::Options dbopt;
  dbopt.lowercase     = true;
  dbopt.min_frequency = min_freq;
  dbopt.position_cap  = 0;
  DictionaryBuilder db(dbopt);
  db.ingest_spans(spans);
  std::vector<DictEntry> entries;
  db.finalize(entries);
  if (suffix)
    DictionaryBuilder::augment_with_suffixes(entries);
  EmbeddingPreprocessor::Options eopt;
  eopt.enable = embed;
  eopt.window = window;
  EmbeddingPreprocessor ep(eopt);
  ep.build_graph(spans);
  std::vector<std::string>                  ordered = ep.reorder_words(entries);
  std::unordered_map<std::string, uint64_t> idmap;
  for (size_t i = 0; i < ordered.size(); ++i)
    idmap[ordered[i]] = (uint64_t)i;
  std::vector<std::pair<std::string, uint64_t>> out;
  for (auto &e : entries)
    out.emplace_back(e.word, idmap[e.word]);
  std::sort(out.begin(), out.end(),
            [](auto &a, auto &b) { return a.first < b.first; });

  Dictionary dict;
  for (auto &kv : out) {
    dict.word_to_id[kv.first]  = kv.second;
    dict.id_to_word[kv.second] = kv.first;
  }
  if (!dict_out.empty())
    Dictionary::Save(dict_out, out);

  Compressor   comp(dict, suffix, prefer_suffix, force_suffix);
  std::string  encoded = comp.compress(spans);
  Decompressor dec(dict);
  std::string  plain = dec.decompress(encoded);

  std::cout << "Compressed:\n" << encoded << "\n\n";
  std::cout << "Roundtrip OK: " << (plain == data ? "YES" : "NO") << "\n";
  if (!outbin.empty())
    save_file(outbin, encoded);
  return 0;
}

// Minimal unit test stubs
static void unit_tests() {
  // base62
  for (uint64_t v : {0ULL, 1ULL, 61ULL, 62ULL, 12345ULL, (1ULL << 40)}) {
    auto     s  = base62_encode(v);
    uint64_t r  = 0;
    bool     ok = base62_decode(s, r);
    if (!ok || r != v)
      std::fprintf(stderr, "base62 fail %s\n", s.c_str());
  }
}

static void print_usage() {
  std::cout << "dictionary-compress\n"
            << "  build-dict   --input file.txt --dict dict.dict [--embed] "
               "[--suffix] [--min-frequency N] [--window-size N]\n"
            << "  compress     --input file.txt --dict dict.dict --output "
               "out.bin [--suffix]\n"
            << "  decompress   --input out.bin --dict dict.dict --output "
               "restored.txt\n"
            << "  test-comp    [--embed] [--suffix] --string \"...\" | --input "
               "file.txt [--dict out.dict] [--output out.bin] [--min-frequency "
               "N] [--window-size N]\n";
}

int main(int argc, char **argv) {
  unit_tests();
  if (argc < 2) {
    print_usage();
    return 0;
  }
  std::string cmd = argv[1];

  std::string input, dict_path, output, test_string, test_input, dict_out,
      outbin;
  bool     embed         = false;
  bool     suffix        = false;
  bool     prefer_suffix = false;
  bool     force_suffix  = false;
  uint64_t min_freq      = 1;
  int      window        = 4;

  auto     get           = [&](int &i) {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value for %s\n", argv[i]);
      std::exit(1);
    }
    return std::string(argv[++i]);
  };
  auto parse_common = [&](int start) {
    for (int i = start; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--input")
        input = get(i);
      else if (a == "--dict")
        dict_path = get(i);
      else if (a == "--output")
        output = get(i);
      else if (a == "--string")
        test_string = get(i);
      else if (a == "--embed")
        embed = true;
      else if (a == "--min-frequency") {
        min_freq = std::strtoull(get(i).c_str(), nullptr, 10);
      } else if (a == "--window-size") {
        window = std::atoi(get(i).c_str());
      } else if (a == "--suffix") {
        suffix = true;
      } else if (a == "--prefer-suffix") {
        prefer_suffix = true;
      } else if (a == "--force-suffix") {
        force_suffix = true;
      } else if (a == "--dict-out" || a == "--dict_out" || a == "--dictout") {
        dict_out = get(i);
      } else {
        std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      }
    }
  };

  if (cmd == "build-dict") {
    parse_common(2);
    if (input.empty() || dict_path.empty()) {
      print_usage();
      return 1;
    }
    return cmd_build_dict(input, dict_path, embed, suffix, min_freq, window);
  } else if (cmd == "compress") {
    parse_common(2);
    if (input.empty() || dict_path.empty()) {
      print_usage();
      return 1;
    }
    return cmd_compress(input, dict_path, output, suffix, prefer_suffix,
                        force_suffix);
  } else if (cmd == "decompress") {
    parse_common(2);
    if (input.empty() || dict_path.empty()) {
      print_usage();
      return 1;
    }
    return cmd_decompress(input, dict_path, output);
  } else if (cmd == "test-comp") {
    parse_common(2);
    return cmd_test(test_string, test_input, dict_out, embed, suffix,
                    prefer_suffix, force_suffix, min_freq, window, output);
  } else {
    print_usage();
    return 1;
  }
}
