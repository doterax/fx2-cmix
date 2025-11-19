#include "text_utils.hpp"

bool is_ascii(uint8_t c) {
  return c < 0x80;
}

bool is_digit(uint8_t c) {
  return c >= '0' && c <= '9';
}

bool is_alpha(uint8_t c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool is_alphanum(uint8_t c) {
  return is_alpha(c) || is_digit(c);
}

size_t utf8_char_len(uint8_t c) {
  if (c < 0x80)
    return 1;
  if ((c & 0xE0) == 0xC0)
    return 2;
  if ((c & 0xF0) == 0xE0)
    return 3;
  if ((c & 0xF8) == 0xF0)
    return 4;
  return 1;
}

size_t encode_varint(uint64_t value, std::vector<uint8_t> &out) {
  size_t count = 0;
  while (value >= 0x80) {
    out.push_back((uint8_t)(value | 0x80));
    value >>= 7;
    count++;
  }
  out.push_back((uint8_t)value);
  return count + 1;
}

size_t decode_varint(const std::vector<uint8_t> &in, size_t &pos,
                     uint64_t &value) {
  value     = 0;
  int shift = 0;
  while (pos < in.size()) {
    uint8_t b = in[pos++];
    value |= (uint64_t)(b & 0x7F) << shift;
    if ((b & 0x80) == 0)
      return shift / 7 + 1;
    shift += 7;
  }
  return 0;
}
