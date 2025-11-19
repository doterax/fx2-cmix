#include "numbers_codec.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
struct BitWriter {
  std::vector<uint8_t> bytes;
  uint8_t              cur = 0;
  int                  bp  = 0;
  void                 putBit(bool b) {
    cur |= uint8_t(b) << bp;
    if (++bp == 8) {
      bytes.push_back(cur);
      cur = 0;
      bp  = 0;
    }
  }
  void putBits(uint64_t v, int n) {
    for (int i = 0; i < n; ++i)
      putBit((v >> i) & 1);
  }
  void flush() {
    if (bp > 0) {
      bytes.push_back(cur);
      cur = 0;
      bp  = 0;
    }
  }
  size_t bitCount() const { return bytes.size() * 8 + bp; }
};
} // namespace

void encode_varint_to(std::vector<uint8_t> &out, uint64_t v) {
  while (true) {
    uint8_t b = uint8_t(v & 0x7F);
    v >>= 7;
    if (v) {
      out.push_back(uint8_t(b | 0x80));
    } else {
      out.push_back(b);
      break;
    }
  }
}

bool decode_varint_from(const std::vector<uint8_t> &buf, size_t start,
                        uint64_t &value, size_t &consumed) {
  value     = 0;
  consumed  = 0;
  int shift = 0;
  while (start + consumed < buf.size()) {
    uint8_t b = buf[start + consumed];
    value |= uint64_t(b & 0x7F) << shift;
    ++consumed;
    if (!(b & 0x80))
      return true;
    shift += 7;
    if (shift > 63)
      return false;
  }
  return false;
}

static std::vector<uint8_t> encode_gamma(const std::vector<uint64_t> &nums) {
  BitWriter bw;
  for (uint64_t n : nums) {
    uint64_t x = n + 1;
    int      k = 64 - __builtin_clzll(x); // bits length of x
    // prefix zeros
    for (int i = 0; i < k - 1; ++i)
      bw.putBit(false);
    // write x in binary (k bits msb->lsb) -> adapt to writer little-endian per
    // bit? We wrote bits LSB-first above. Need to emit bits LSB-first to match
    // putBits; convert.
    for (int i = 0; i < k; ++i) {
      bool bit = (x >> (k - 1 - i)) & 1;
      bw.putBit(bit);
    }
  }
  bw.flush();
  return bw.bytes;
}

std::vector<uint8_t> encode_numbers_gamma(const std::vector<uint64_t>& numbers) {
  return encode_gamma(numbers);
}

bool decode_numbers_gamma(const std::vector<uint8_t>& encoded, uint64_t count, std::vector<uint64_t>& out) {
  out.clear();
  size_t bpByte = 0; int bpBit = 0;
  auto readBit = [&]() {
    if (bpByte >= encoded.size()) return 0u;
    unsigned b = (encoded[bpByte] >> bpBit) & 1u;
    if (++bpBit == 8) { bpBit = 0; ++bpByte; }
    return b;
  };
  for (uint64_t i = 0; i < count; ++i) {
    int zeros = 0;
    while (true) {
      if (bpByte >= encoded.size()) return false;
      if (readBit() == 0) zeros++; else break;
    }
    int k = zeros + 1;
    uint64_t x = 1;
    for (int b = 0; b < k - 1; ++b) {
      if (bpByte >= encoded.size()) return false;
      x = (x << 1) | readBit();
    }
    uint64_t n = x - 1;
    out.push_back(n);
  }
  return true;
}
