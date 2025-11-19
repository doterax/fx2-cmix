#include "chunked_preprocessor_v2.hpp"
#include "text_utils.hpp"
#include <algorithm>
#include <cstdio> // retained for legacy includes (can be removed if no FILE* left)
#include <string>
#include <vector>

// Region types (3 bits)
enum ChunkTypeV2 {
  CHUNK_RAW            = 0,
  CHUNK_ASCII_SENTENCE = 1,
  CHUNK_BRACKETS       = 2,
  CHUNK_WIKI_HEADER    = 3,
  CHUNK_XML_TAG        = 4,
  CHUNK_CURLY_BRACKETS = 5,
  CHUNK_UTF8_RUN       = 6,
  CHUNK_RESERVED       = 7
};

// Transformations (2 bits)
enum TransformTypeV2 {
  TRANSFORM_NONE        = 0,
  TRANSFORM_FIRST_UPPER = 1,
  TRANSFORM_ALL_UPPER   = 2,
  TRANSFORM_RESERVED    = 3
};

static void write_type_and_transform(BitStreamWriter &control, uint8_t type,
                                     uint8_t transform) {
  control.writeBits(type, 3);
  control.writeBits(transform, 2);
}

static void write_varint_bits(BitStreamWriter &control, uint64_t size) {
  std::vector<uint8_t> tmp;
  encode_varint(size, tmp);
  for (uint8_t b : tmp)
    control.writeBits(b, 8);
}

static bool read_varint_bits(BitStreamReader &control, uint64_t &value) {
  value     = 0;
  int shift = 0;
  while (true) {
    if (control.eof())
      return false;
    uint64_t b = control.readBits(8);
    value |= (uint64_t)(b & 0x7F) << shift;
    if ((b & 0x80) == 0)
      return true;
    shift += 7;
    if (shift > 63)
      return false;
  }
}

static bool is_all_upper(const std::vector<uint8_t> &buf, size_t start,
                         size_t end) {
  bool any_alpha = false;
  for (size_t i = start; i < end; ++i) {
    uint8_t c = buf[i];
    if (is_alpha(c)) {
      any_alpha = true;
      if (!(c >= 'A' && c <= 'Z'))
        return false;
    }
  }
  return any_alpha;
}
static bool is_first_upper(const std::vector<uint8_t> &buf, size_t start,
                           size_t end) {
  bool firstFound = false;
  for (size_t i = start; i < end; ++i) {
    uint8_t c = buf[i];
    if (is_alpha(c)) {
      if (!firstFound) {
        if (!(c >= 'A' && c <= 'Z'))
          return false; // first alpha not uppercase
        firstFound = true;
      } else {
        // subsequent alpha must not be uppercase for FIRST_UPPER pattern
        if (c >= 'A' && c <= 'Z')
          return false;
      }
    }
  }
  return firstFound; // at least one alpha and pattern satisfied
}
static void to_lowercase_store(const std::vector<uint8_t> &buf, size_t start,
                               size_t end, std::vector<uint8_t> &out) {
  out.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    uint8_t c = buf[i];
    if (c >= 'A' && c <= 'Z')
      c = uint8_t(c - 'A' + 'a');
    out.push_back(c);
  }
}

static bool name_in_whitelist(const std::vector<std::string> &wl,
                              const std::vector<uint8_t> &buf, size_t start,
                              size_t end) {
  for (const auto &s : wl) {
    if (s.size() == end - start &&
        std::equal(s.begin(), s.end(), buf.begin() + start))
      return true;
  }
  return false;
}

// Stream variant: read all remaining bytes from input BitStreamReader as raw
// bytes
bool ChunkedPreprocessorV2::compress_stream(BitStreamReader       &input,
                                            BitStreamWriter       &control,
                                            BitStreamWriter       &ascii,
                                            BitStreamWriter       &utf8,
                                            const ChunkedV2Config &cfg) {
  std::vector<uint8_t> buffer;
  buffer.reserve(1024);
  while (!input.eof()) {
    uint64_t b = input.readBits(8);
    buffer.push_back((uint8_t)b);
  }
  size_t pos = 0;
  while (pos < buffer.size()) {
    uint8_t              type         = CHUNK_RAW;
    uint8_t              transform    = TRANSFORM_NONE;
    size_t               region_start = pos;
    size_t               region_end   = pos + 1;
    std::vector<uint8_t> payload;
    bool                 utf8_region = false;
    uint8_t              c           = buffer[pos];
    if (!is_ascii(c)) {
      size_t i = pos;
      while (i < buffer.size() && !is_ascii(buffer[i])) {
        size_t len = utf8_char_len(buffer[i]);
        if (i + len > buffer.size())
          len = 1;
        i += len;
      }
      type        = CHUNK_UTF8_RUN;
      region_end  = i;
      utf8_region = true;
    } else {
      // (Repeat pattern logic identical to file-based compress)
      if (c == '[' && pos + 3 < buffer.size() && buffer[pos + 1] == '[') {
        size_t inner_start = pos + 2;
        size_t i           = inner_start;
        bool   aborted     = false;
        while (i + 1 < buffer.size() &&
               !(buffer[i] == ']' && buffer[i + 1] == ']')) {
          if ((i - inner_start) >= cfg.brackets_max_span) { aborted = true; break; }
          i++;
        }
        if (!aborted && i + 1 < buffer.size()) {
          type       = CHUNK_BRACKETS;
          region_end = i + 2;
          if (is_all_upper(buffer, inner_start, i))
            transform = TRANSFORM_ALL_UPPER;
          else if (is_first_upper(buffer, inner_start, i))
            transform = TRANSFORM_FIRST_UPPER;
          if (transform == TRANSFORM_NONE) {
            for (size_t k = inner_start; k < i; ++k)
              payload.push_back(buffer[k]);
          } else {
            to_lowercase_store(buffer, inner_start, i, payload);
          }
        }
      }
      if (type == CHUNK_RAW && c == '=' && pos + 3 < buffer.size() &&
          buffer[pos + 1] == '=') {
        size_t inner_start = pos + 2;
        size_t i           = inner_start;
        bool   aborted     = false;
        while (i + 1 < buffer.size() &&
               !(buffer[i] == '=' && buffer[i + 1] == '=')) {
          if ((i - inner_start) >= cfg.wiki_header_max_span) { aborted = true; break; }
          i++;
        }
        if (!aborted && i + 1 < buffer.size()) {
          type       = CHUNK_WIKI_HEADER;
          region_end = i + 2;
          if (is_all_upper(buffer, inner_start, i))
            transform = TRANSFORM_ALL_UPPER;
          else if (is_first_upper(buffer, inner_start, i))
            transform = TRANSFORM_FIRST_UPPER;
          if (transform == TRANSFORM_NONE) {
            for (size_t k = inner_start; k < i; ++k)
              payload.push_back(buffer[k]);
          } else {
            to_lowercase_store(buffer, inner_start, i, payload);
          }
        }
      }
      if (type == CHUNK_RAW && c == '{' && pos + 3 < buffer.size() &&
          buffer[pos + 1] == '{') {
        size_t inner_start = pos + 2;
        size_t i           = inner_start;
        bool   aborted     = false;
        while (i + 1 < buffer.size() &&
               !(buffer[i] == '}' && buffer[i + 1] == '}')) {
          if ((i - inner_start) >= cfg.curly_brackets_max_span) { aborted = true; break; }
          i++;
        }
        if (!aborted && i + 1 < buffer.size()) {
          type       = CHUNK_CURLY_BRACKETS;
          region_end = i + 2;
          if (is_all_upper(buffer, inner_start, i))
            transform = TRANSFORM_ALL_UPPER;
          else if (is_first_upper(buffer, inner_start, i))
            transform = TRANSFORM_FIRST_UPPER;
          if (transform == TRANSFORM_NONE) {
            for (size_t k = inner_start; k < i; ++k)
              payload.push_back(buffer[k]);
          } else {
            to_lowercase_store(buffer, inner_start, i, payload);
          }
        }
      }
      if (type == CHUNK_RAW && c == '<' && pos + 2 < buffer.size() &&
          is_alpha(buffer[pos + 1])) {
        size_t name_start = pos + 1;
        size_t name_end   = name_start;
        while (name_end < buffer.size() && is_alpha(buffer[name_end]))
          name_end++;
        bool   allowed      = false;
        size_t name_len_chk = name_end - name_start;
        if (cfg.xml_lowercase_rule) {
          if (name_len_chk > 0 && name_len_chk <= cfg.xml_max_name_len) {
            allowed = true;
            for (size_t k = name_start; k < name_end; ++k) {
              uint8_t ch2 = buffer[k];
              if (!(ch2 >= 'a' && ch2 <= 'z')) {
                allowed = false;
                break;
              }
            }
          }
        } else {
          if (!cfg.xml_whitelist.empty()) {
            allowed = name_in_whitelist(cfg.xml_whitelist, buffer, name_start,
                                        name_end);
          }
        }
        if (allowed) {
          size_t open_end = name_end;
          bool   found_gt = false;
          while (open_end < buffer.size()) {
            if (buffer[open_end] == '>') {
              found_gt = true;
              break;
            }
            open_end++;
          }
          if (found_gt) {
            size_t attrs_start   = name_end;
            size_t attrs_end     = open_end;
            size_t content_start = open_end + 1;
            size_t i             = content_start;
            size_t name_len      = name_end - name_start;
            bool   matched       = false;
            bool   aborted       = false;
            while (i + name_len + 3 <= buffer.size()) {
              if ((i - content_start) > cfg.xml_max_span) {
                aborted = true;
                break;
              }
              if (buffer[i] == '<' && i + 1 < buffer.size() &&
                  buffer[i + 1] != '/' && is_alpha(buffer[i + 1])) {
                aborted = true;
                break;
              }
              if (cfg.xml_ascii_only_content && !is_ascii(buffer[i])) {
                aborted = true;
                break;
              }
              if (buffer[i] == '<' && buffer[i + 1] == '/' &&
                  std::equal(buffer.begin() + i + 2,
                             buffer.begin() + i + 2 + name_len,
                             buffer.begin() + name_start) &&
                  buffer[i + 2 + name_len] == '>') {
                size_t content_end = i;
                type               = CHUNK_XML_TAG;
                region_end         = i + 3 + name_len;
                if (is_all_upper(buffer, content_start, content_end))
                  transform = TRANSFORM_ALL_UPPER;
                else if (is_first_upper(buffer, content_start, content_end))
                  transform = TRANSFORM_FIRST_UPPER;
                for (size_t k = name_start; k < name_end; ++k)
                  payload.push_back(buffer[k]);
                payload.push_back(0);
                for (size_t k = attrs_start; k < attrs_end; ++k)
                  payload.push_back(buffer[k]);
                payload.push_back(0);
                if (transform == TRANSFORM_NONE) {
                  for (size_t k = content_start; k < content_end; ++k)
                    payload.push_back(buffer[k]);
                } else {
                  to_lowercase_store(buffer, content_start, content_end,
                                     payload);
                }
                matched = true;
                break;
              }
              i++;
            }
            (void)matched;
            (void)aborted;
          }
        }
      }
      if (type == CHUNK_RAW && is_alpha(c) && c >= 'A' && c <= 'Z') {
        // DEBUG (disabled): probe ASCII_SENTENCE scanning for small tests
        // fprintf(stderr, "ASCII start pos=%zu char=%c\n", (size_t)pos, (char)c);
        size_t i     = pos;
        // fprintf(stderr, "pre while i=%zu size=%zu\n", (size_t)i, (size_t)buffer.size());
        bool   ended = false;
        while (i < buffer.size()) {
          uint8_t cc = buffer[i];
          if (cc == '.' || cc == '!' || cc == '?') {
            // fprintf(stderr, "ASCII end punct %c at %zu\n", (char)cc, (size_t)i);
            ended = true;
            i++;
            break;
          }
          if (!is_ascii(cc)) break;
          bool allowed = (is_alphanum(cc) || cc == ' ' || cc == ',' || cc == ';' ||
                          cc == ':' || cc == '\'' || cc == '"' || cc == '(' ||
                          cc == ')' || cc == '&' || cc == '-' || cc == '/');
          if (!allowed) {
            // fprintf(stderr, "ASCII blocked 0x%02X '%c' at %zu\n", cc, (char)cc, (size_t)i);
            break;
          }
          if ((i - pos) >= cfg.ascii_sentence_max_span) break;
          i++;
        }
        // fprintf(stderr, "ASCII loop end i=%zu ended=%d size=%zu\n", (size_t)i, (int)ended, (size_t)buffer.size());
        if (ended) {
          type       = CHUNK_ASCII_SENTENCE;
          region_end = i;
          if (is_all_upper(buffer, pos, region_end - 1))
            transform = TRANSFORM_ALL_UPPER;
          else if (is_first_upper(buffer, pos, region_end - 1))
            transform = TRANSFORM_FIRST_UPPER;
          if (transform == TRANSFORM_NONE) {
            for (size_t k = pos; k < region_end - 1; ++k)
              payload.push_back(buffer[k]);
          } else {
            to_lowercase_store(buffer, pos, region_end - 1, payload);
          }
        }
      }
      if (type == CHUNK_RAW) {
        size_t i = pos;
        while (i < buffer.size()) {
          uint8_t ch = buffer[i];
          if (!is_ascii(ch)) break;
          bool boundary = false;
          if (ch == '[' && i + 1 < buffer.size() && buffer[i + 1] == '[') boundary = true;
          else if (ch == '=' && i + 1 < buffer.size() && buffer[i + 1] == '=') boundary = true;
          else if (ch == '{' && i + 1 < buffer.size() && buffer[i + 1] == '{') boundary = true;
          else if (ch == '<' && i + 1 < buffer.size() && is_alpha(buffer[i + 1])) boundary = true;
          else if (is_alpha(ch) && ch >= 'A' && ch <= 'Z' && i == pos) boundary = true; // uppercase boundary only at start
          if (boundary && i != pos) break;
          i++;
        }
        region_end = i;
      }
    }
    write_type_and_transform(control, type, transform);
    size_t payload_len = 0;
    if (utf8_region)
      payload_len = region_end - region_start;
    else if (type == CHUNK_ASCII_SENTENCE)
      payload_len = payload.size();
    else if (type == CHUNK_BRACKETS || type == CHUNK_WIKI_HEADER ||
             type == CHUNK_CURLY_BRACKETS || type == CHUNK_XML_TAG)
      payload_len = payload.size();
    else
      payload_len = region_end - region_start;
    write_varint_bits(control, (uint64_t)payload_len);
    if (utf8_region) {
      for (size_t i2 = region_start; i2 < region_end; ++i2)
        utf8.writeBits(buffer[i2], 8);
    } else if (type == CHUNK_ASCII_SENTENCE || type == CHUNK_BRACKETS ||
               type == CHUNK_WIKI_HEADER || type == CHUNK_CURLY_BRACKETS ||
               type == CHUNK_XML_TAG) {
      for (auto b : payload)
        ascii.writeBits(b, 8);
    } else {
      for (size_t i2 = region_start; i2 < region_end; ++i2)
        ascii.writeBits(buffer[i2], 8);
    }
    pos = region_end;
  }
  control.flush();
  ascii.flush();
  utf8.flush();
  return true;
}

// Stream variant: write reconstructed bytes to output BitStreamWriter
bool ChunkedPreprocessorV2::decompress_stream(BitStreamReader &control,
                                              BitStreamReader &ascii,
                                              BitStreamReader &utf8,
                                              BitStreamWriter &output,
                                              const ChunkedV2Config &) {
  while (!control.eof()) {
    if (control.eof())
      break;
    uint64_t type      = control.readBits(3);
    uint64_t transform = control.readBits(2);
    uint64_t len       = 0;
    if (!read_varint_bits(control, len))
      break;
    if (type == CHUNK_UTF8_RUN) {
      for (uint64_t i = 0; i < len; ++i) {
        if (utf8.eof())
          return false;
        uint64_t b = utf8.readBits(8);
        output.writeBits(b, 8);
      }
    } else {
      std::vector<uint8_t> temp;
      temp.reserve((size_t)len);
      for (uint64_t i = 0; i < len; ++i) {
        if (ascii.eof())
          return false;
        uint64_t b = ascii.readBits(8);
        temp.push_back((uint8_t)b);
      }
      std::vector<uint8_t> outbuf;
      if (type == CHUNK_BRACKETS || type == CHUNK_WIKI_HEADER ||
          type == CHUNK_CURLY_BRACKETS) {
        // restore wrapper after transform
        if (transform == TRANSFORM_ALL_UPPER) {
          for (auto &ch : temp)
            if (ch >= 'a' && ch <= 'z')
              ch = uint8_t(ch - 'a' + 'A');
        } else if (transform == TRANSFORM_FIRST_UPPER) {
          for (size_t i = 0; i < temp.size(); ++i) {
            if (is_alpha(temp[i])) {
              if (temp[i] >= 'a' && temp[i] <= 'z')
                temp[i] = uint8_t(temp[i] - 'a' + 'A');
              break;
            }
          }
        }
        if (type == CHUNK_BRACKETS) {
          outbuf.push_back('[');
          outbuf.push_back('[');
          for (auto b : temp)
            outbuf.push_back(b);
          outbuf.push_back(']');
          outbuf.push_back(']');
        } else if (type == CHUNK_WIKI_HEADER) {
          outbuf.push_back('=');
          outbuf.push_back('=');
          for (auto b : temp)
            outbuf.push_back(b);
          outbuf.push_back('=');
          outbuf.push_back('=');
        } else {
          outbuf.push_back('{');
          outbuf.push_back('{');
          for (auto b : temp)
            outbuf.push_back(b);
          outbuf.push_back('}');
          outbuf.push_back('}');
        }
      } else if (type == CHUNK_XML_TAG) {
        size_t sep1 = 0;
        while (sep1 < temp.size() && temp[sep1] != 0)
          sep1++;
        std::string tagName;
        for (size_t i = 0; i < sep1; ++i)
          tagName.push_back((char)temp[i]);
        size_t sep2 = sep1 + 1;
        while (sep2 < temp.size() && temp[sep2] != 0)
          sep2++;
        std::string attrs;
        if (sep1 < temp.size() && sep1 < sep2 && sep2 <= temp.size()) {
          for (size_t i = sep1 + 1; i < sep2; ++i)
            attrs.push_back((char)temp[i]);
        }
        std::vector<uint8_t> content;
        if (sep2 < temp.size() && temp[sep2] == 0) {
          for (size_t i = sep2 + 1; i < temp.size(); ++i)
            content.push_back(temp[i]);
        }
        if (transform == TRANSFORM_ALL_UPPER) {
          for (auto &ch : content)
            if (ch >= 'a' && ch <= 'z')
              ch = uint8_t(ch - 'a' + 'A');
        } else if (transform == TRANSFORM_FIRST_UPPER) {
          for (size_t i = 0; i < content.size(); ++i) {
            if (is_alpha(content[i])) {
              if (content[i] >= 'a' && content[i] <= 'z')
                content[i] = uint8_t(content[i] - 'a' + 'A');
              break;
            }
          }
        }
        outbuf.push_back('<');
        for (auto ch : tagName)
          outbuf.push_back((uint8_t)ch);
        for (auto ch : attrs)
          outbuf.push_back((uint8_t)ch);
        outbuf.push_back('>');
        for (auto b : content)
          outbuf.push_back(b);
        outbuf.push_back('<');
        outbuf.push_back('/');
        for (auto ch : tagName)
          outbuf.push_back((uint8_t)ch);
        outbuf.push_back('>');
      } else if (type == CHUNK_ASCII_SENTENCE) {
        if (transform == TRANSFORM_ALL_UPPER) {
          for (auto &ch : temp)
            if (ch >= 'a' && ch <= 'z')
              ch = uint8_t(ch - 'a' + 'A');
        } else if (transform == TRANSFORM_FIRST_UPPER) {
          for (size_t i = 0; i < temp.size(); ++i) {
            if (is_alpha(temp[i])) {
              if (temp[i] >= 'a' && temp[i] <= 'z')
                temp[i] = uint8_t(temp[i] - 'a' + 'A');
              break;
            }
          }
        }
        for (auto b : temp)
          outbuf.push_back(b);
        outbuf.push_back('.');
      } else {
        // RAW or others
        if (transform == TRANSFORM_ALL_UPPER) {
          for (auto &ch : temp)
            if (ch >= 'a' && ch <= 'z')
              ch = uint8_t(ch - 'a' + 'A');
        } else if (transform == TRANSFORM_FIRST_UPPER) {
          for (size_t i = 0; i < temp.size(); ++i) {
            if (is_alpha(temp[i])) {
              if (temp[i] >= 'a' && temp[i] <= 'z')
                temp[i] = uint8_t(temp[i] - 'a' + 'A');
              break;
            }
          }
        }
        outbuf.insert(outbuf.end(), temp.begin(), temp.end());
      }
      for (auto b : outbuf)
        output.writeBits(b, 8);
    }
  }
  output.flush();
  return true;
}
