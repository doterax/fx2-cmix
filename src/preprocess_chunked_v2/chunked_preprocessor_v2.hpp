#pragma once
#include "bitstream.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct ChunkedV2Config {
  // XML tag detection settings
  // If true, chunk any XML tag whose name is lowercase ASCII and length <=
  // xml_max_name_len. If false, use xml_whitelist for allowed tag names.
  bool                     xml_lowercase_rule = true;
  uint32_t                 xml_max_name_len   = 64;
  std::vector<std::string> xml_whitelist; // e.g., {"title","comment","text"}
  // Maximum allowed distance (content bytes) to closing tag before aborting XML
  // chunk (fallback RAW)
  uint32_t xml_max_span = 1024; // bytes of content scan limit
  // If true, abort XML chunk if non-ASCII appears in content (so UTF8_RUN
  // handles it)
  bool xml_ascii_only_content = true;

  // Anti-greed limits for other wrappers (payload/content byte caps)
  // Maximum number of inner bytes allowed for [[...]] before aborting
  uint32_t brackets_max_span      = 512;
  // Maximum number of inner bytes allowed for ==...== before aborting
  uint32_t wiki_header_max_span   = 512;
  // Maximum number of inner bytes allowed for {{...}} before aborting
  uint32_t curly_brackets_max_span = 512;

  // Minimum digit run length to treat as NUMBER chunk (positive integer)
  uint32_t min_number_len = 1;

  // Do not split a current non-empty RAW context on encountering a NUMBER run
  // unless at least this many bytes have already accumulated in the RAW
  // context. This prevents fragmentation by short contexts.
  uint32_t min_raw_context_before_number_split = 64;
};

// Unified context structs for (conceptual v3) compression/decompression
struct ChunkedV3CompressionContext {
  BitStreamReader       *input = nullptr;
  BitStreamWriter       *control = nullptr;
  BitStreamWriter       *ascii = nullptr;
  BitStreamWriter       *utf8 = nullptr;
  // Collected numbers: interleaved (chunk lengths + NUMBER values) FIFO
  std::vector<uint64_t>  numbers;
};

struct ChunkedV3DecompressionContext {
  BitStreamReader       *control = nullptr;
  BitStreamReader       *ascii = nullptr;
  BitStreamReader       *utf8 = nullptr;
  BitStreamWriter       *output = nullptr;
  std::vector<uint64_t>  numbers; // pre-decoded number list
  size_t                 number_pos = 0; // consumption index
};

class ChunkedPreprocessorV2 {
public:
  // Original interface retained (will internally use v3 context logic for NUMBER handling)
  static bool compress_stream(BitStreamReader &input, BitStreamWriter &control,
                              BitStreamWriter &ascii, BitStreamWriter &utf8,
                              const ChunkedV2Config &cfg, std::vector<uint64_t> *numbers_out = nullptr);
  static bool decompress_stream(BitStreamReader &control,
                                BitStreamReader &ascii, BitStreamReader &utf8,
                                BitStreamWriter       &output,
                                const ChunkedV2Config &cfg,
                                const std::vector<uint64_t> *numbers_in = nullptr);
};
