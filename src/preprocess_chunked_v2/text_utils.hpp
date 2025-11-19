#pragma once
#include <cstdint>
#include <string>
#include <vector>

bool   is_ascii(uint8_t c);
bool   is_digit(uint8_t c);
bool   is_alpha(uint8_t c);
bool   is_alphanum(uint8_t c);
size_t utf8_char_len(uint8_t c);
size_t encode_varint(uint64_t value, std::vector<uint8_t> &out);
size_t decode_varint(const std::vector<uint8_t> &in, size_t &pos,
                     uint64_t &value);
