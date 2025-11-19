#pragma once
#include <cstdint>
#include <vector>
#include <string>

// Number encoding: only Elias Gamma (stores n+1 to handle zeros)

// Build gamma bitstream for numbers list (returns byte vector)
std::vector<uint8_t> encode_numbers_gamma(const std::vector<uint64_t>& numbers);

// Decode gamma bitstream given count; returns false if malformed
bool decode_numbers_gamma(const std::vector<uint8_t>& encoded, uint64_t count, std::vector<uint64_t>& out);

// Decode sidecar file contents into numbers list.
// Sidecar format:
//   byte: method id
//   varint: param (present if method != 0)
//   varint: count
//   bitstream bytes: encoded numbers
// Returns true on success; fills numbers_out.
bool decode_numbers_sidecar(const std::vector<uint8_t>& sidecar, std::vector<uint64_t>& numbers_out, uint8_t &method, uint64_t &param);

// Utility: encode varint append to vector.
void encode_varint_to(std::vector<uint8_t>& out, uint64_t v);

// Utility: decode varint from buffer starting at pos; returns false if fails.
bool decode_varint_from(const std::vector<uint8_t>& buf, size_t start, uint64_t &value, size_t &consumed);

// (No estimation or alternative methods retained; gamma only)
