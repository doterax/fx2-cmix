#include "bitstream.hpp"
#include <cstdint>
#include <vector>

class BitStreamWriterVector : public BitStreamWriter {
  std::vector<uint8_t> &buffer;
  uint8_t               curByte = 0;
  int                   bitPos  = 0;

public:
  BitStreamWriterVector(std::vector<uint8_t> &buf) : buffer(buf) {}
  void writeBit(bool bit) override {
    curByte |= (bit ? 1 : 0) << bitPos;
    bitPos++;
    if (bitPos == 8) {
      buffer.push_back(curByte);
      curByte = 0;
      bitPos  = 0;
    }
  }
  void writeBits(uint64_t value, size_t count) override {
    for (size_t i = 0; i < count; ++i)
      writeBit((value >> i) & 1);
  }
  void flush() override {
    if (bitPos > 0)
      buffer.push_back(curByte);
    curByte = 0;
    bitPos  = 0;
  }
};

class BitStreamReaderVector : public BitStreamReader {
  const std::vector<uint8_t> &buffer;
  size_t                      bytePos = 0;
  int                         bitPos  = 0;

public:
  BitStreamReaderVector(const std::vector<uint8_t> &buf) : buffer(buf) {}
  bool readBit() override {
    if (bytePos >= buffer.size())
      return false;
    bool bit = (buffer[bytePos] >> bitPos) & 1;
    bitPos++;
    if (bitPos == 8) {
      bitPos = 0;
      bytePos++;
    }
    return bit;
  }
  uint64_t readBits(size_t count) override {
    uint64_t v = 0;
    for (size_t i = 0; i < count; ++i)
      v |= (uint64_t(readBit()) << i);
    return v;
  }
  bool eof() const override { return bytePos >= buffer.size(); }
};
