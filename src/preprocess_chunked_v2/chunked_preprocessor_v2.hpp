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

  // ASCII sentence detection cap (bytes before terminator)
  uint32_t ascii_sentence_max_span = 512;
};

class ChunkedPreprocessorV2 {
public:
  // Stream-based API only (FILE* removed)
  static bool compress_stream(BitStreamReader &input, BitStreamWriter &control,
                              BitStreamWriter &ascii, BitStreamWriter &utf8,
                              const ChunkedV2Config &cfg);
  static bool decompress_stream(BitStreamReader &control,
                                BitStreamReader &ascii, BitStreamReader &utf8,
                                BitStreamWriter       &output,
                                const ChunkedV2Config &cfg);
};
