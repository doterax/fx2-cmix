#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "numbers_codec.hpp"

static bool read_varint_file(FILE *f, uint64_t &v) {
  v         = 0;
  int shift = 0;
  while (true) {
    int b = fgetc(f);
    if (b == EOF)
      return false;
    v |= (uint64_t)(b & 0x7F) << shift;
    if ((b & 0x80) == 0)
      return true;
    shift += 7;
    if (shift > 63)
      return false;
  }
}

class BitStreamReaderVector {
  const std::vector<uint8_t> &buf;
  size_t                      bp  = 0;
  int                         bit = 0;

public:
  BitStreamReaderVector(const std::vector<uint8_t> &b) : buf(b) {}
  bool     eof() const { return bp >= buf.size(); }
  uint64_t readBits(size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
      if (bp >= buf.size())
        return v;
      v |= ((uint64_t)((buf[bp] >> bit) & 1)) << i;
      if (++bit == 8) {
        bit = 0;
        ++bp;
      }
    }
    return v;
  }
  uint64_t consumedBits() const { return uint64_t(bp) * 8 + uint64_t(bit); }
};

struct Stat {
  uint64_t count = 0, bytes = 0, controlBytes = 0, minLen = UINT64_MAX,
           maxLen = 0;
};

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <container.v2>" << std::endl;
    return 1;
  }
  const char *path = argv[1];
  FILE       *f    = fopen(path, "rb");
  if (!f) {
    std::cerr << "open failed" << std::endl;
    return 1;
  }
  uint64_t csz = 0, asz = 0, usz = 0, nsz = 0, ncount = 0;
  if (!read_varint_file(f, csz) || !read_varint_file(f, asz) ||
      !read_varint_file(f, usz) || !read_varint_file(f, nsz) || !read_varint_file(f, ncount)) {
    fclose(f);
    std::cerr << "header failed" << std::endl;
    return 1;
  }
  std::vector<uint8_t> control(csz), ascii(asz), utf8(usz), numbersBytes(nsz);
  if (csz && fread(control.data(), 1, csz, f) != csz) {
    fclose(f);
    std::cerr << "read control failed" << std::endl;
    return 1;
  }
  if (asz && fread(ascii.data(), 1, asz, f) != asz) {
    fclose(f);
    std::cerr << "read ascii failed" << std::endl;
    return 1;
  }
  if (usz && fread(utf8.data(), 1, usz, f) != usz) {
    fclose(f);
    std::cerr << "read utf8 failed" << std::endl;
    return 1;
  }
  if (nsz && fread(numbersBytes.data(), 1, nsz, f) != nsz) {
    fclose(f);
    std::cerr << "read numbers failed" << std::endl;
    return 1;
  }
  fclose(f);
  std::cout << "control=" << csz << " ascii=" << asz << " utf8=" << usz << " numbers_bytes=" << nsz << " numbers_count=" << ncount << "\n";
  std::vector<uint64_t> numbers;
  if (!decode_numbers_gamma(numbersBytes, ncount, numbers) || numbers.size()!=ncount) {
    std::cerr << "numbers gamma decode failed" << std::endl;
    return 1;
  }
  // Verify roundtrip of gamma encoding
  auto reenc = encode_numbers_gamma(numbers);
  bool gamma_ok = (reenc.size()==numbersBytes.size() && std::equal(reenc.begin(), reenc.end(), numbersBytes.begin()));
  std::cout << (gamma_ok?"numbers_gamma_roundtrip=OK":"numbers_gamma_roundtrip=FAIL") << "\n";

  BitStreamReaderVector cr(control);
  Stat                  stats[8];
  uint64_t              transformCount[4] = {0, 0, 0, 0};
  uint64_t              totalChunks = 0, totalBytes = 0;

  size_t number_index = 0;
  while (!cr.eof() && number_index < numbers.size()) {
    uint64_t type      = cr.readBits(3);
    uint64_t transform = cr.readBits(2);
    uint64_t len = 0;
    if (type == 1) { // NUMBER chunk consumes two entries: run_len then value
      if (number_index + 1 >= numbers.size()) break; // malformed
      uint64_t run_len = numbers[number_index++];
      uint64_t value   = numbers[number_index++];
      (void)run_len; (void)value; // no bytes accounted (digits reconstructed virtually)
      len = 0;
    } else {
      if (number_index >= numbers.size()) break; // malformed
      len = numbers[number_index++];
    }
    uint64_t headerBits  = 5; // fixed per chunk
    uint64_t headerBytes = (headerBits + 7) / 8; // 1 byte accounting
    if (type > 7)
      type = 7;
    if (transform > 3)
      transform = 3;
    Stat &s = stats[type];
    s.count++;
    s.bytes += len;
    s.controlBytes += headerBytes;
    if (len < s.minLen)
      s.minLen = len;
    if (len > s.maxLen)
      s.maxLen = len;
    transformCount[transform]++;
    totalChunks++;
    totalBytes += len;
  }

  struct Row {
    int      type;
    uint64_t count;
    uint64_t bytes;
    uint64_t ctrlBytes;
    uint64_t minLen;
    uint64_t maxLen;
    double   avg;
  };
  std::vector<Row> rows;
  for (int t = 0; t < 8; ++t)
    if (stats[t].count > 0) {
      rows.push_back(
          {t, stats[t].count, stats[t].bytes, stats[t].controlBytes,
           stats[t].minLen == UINT64_MAX ? 0 : stats[t].minLen, stats[t].maxLen,
           stats[t].count ? double(stats[t].bytes) / double(stats[t].count)
                          : 0.0});
    }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
    if (a.count != b.count)
      return a.count > b.count;
    return a.bytes > b.bytes;
  });

  auto typeName = [](int t) {
    switch (t) {
    case 0:
      return "RAW";
    case 1:
      return "NUMBER";
    case 2:
      return "BRACKETS";
    case 3:
      return "WIKI_HEADER";
    case 4:
      return "XML_TAG";
    case 5:
      return "CURLY_BRACKETS";
    case 6:
      return "UTF8_RUN";
    case 7:
      return "RESERVED";
    default:
      return "UNKNOWN";
    }
  };

  std::cout << "chunks=" << totalChunks << " data_total=" << totalBytes
            << " (ascii+utf8=" << (asz + usz) << ")\n";
  std::cout << "\nNumbers: gamma_bytes=" << nsz << " values=" << numbers.size() << "\n";
  // Table header
  std::cout << "\nChunk Statistics:\n";
  std::cout << "  " << std::left << std::setw(3) << "ID" << std::setw(16)
            << "TYPE" << std::right << std::setw(12) << "COUNT" << std::setw(14)
            << "BYTES" << std::setw(14) << "CTRL_BYTES" << std::setw(12)
            << "AVG" << std::setw(10) << "MIN" << std::setw(10) << "MAX"
            << "\n";
  std::cout << "  " << std::string(3 + 16 + 12 + 14 + 14 + 12 + 10 + 10, '-')
            << "\n";
  for (auto &r : rows) {
    std::cout << "  " << std::left << std::setw(3) << r.type << std::setw(16)
              << typeName(r.type) << std::right << std::setw(12) << r.count
              << std::setw(14) << r.bytes << std::setw(14) << r.ctrlBytes
              << std::setw(12) << std::fixed << std::setprecision(3) << r.avg
              << std::setw(10) << r.minLen << std::setw(10) << r.maxLen << "\n";
  }
  std::cout << "\nTransform Counts:\n";
  std::cout << "  NONE        : " << transformCount[0] << "\n";
  std::cout << "  FIRST_UPPER : " << transformCount[1] << "\n";
  std::cout << "  ALL_UPPER   : " << transformCount[2] << "\n";
  std::cout << "  RESERVED    : " << transformCount[3] << "\n";

  return 0;
}
