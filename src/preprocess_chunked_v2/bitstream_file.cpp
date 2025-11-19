#include "bitstream.hpp"
#include <cstdint>
#include <cstdio>

class BitStreamWriterFile : public BitStreamWriter {
  FILE   *f;
  uint8_t curByte = 0;
  int     bitPos  = 0;

public:
  BitStreamWriterFile(FILE *file) : f(file) {}
  void writeBit(bool bit) override {
    curByte |= (bit ? 1 : 0) << bitPos;
    bitPos++;
    if (bitPos == 8) {
      fputc(curByte, f);
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
      fputc(curByte, f);
    curByte = 0;
    bitPos  = 0;
  }
};

class BitStreamReaderFile : public BitStreamReader {
  FILE   *f;
  uint8_t curByte = 0;
  int     bitPos  = 8;
  bool    atEOF   = false;

public:
  BitStreamReaderFile(FILE *file) : f(file) {}
  bool readBit() override {
    if (bitPos == 8) {
      int b = fgetc(f);
      if (b == EOF) {
        atEOF = true;
        return false;
      }
      curByte = (uint8_t)b;
      bitPos  = 0;
    }
    bool bit = (curByte >> bitPos) & 1;
    bitPos++;
    return bit;
  }
  uint64_t readBits(size_t count) override {
    uint64_t v = 0;
    for (size_t i = 0; i < count; ++i)
      v |= (uint64_t(readBit()) << i);
    return v;
  }
  bool eof() const override { return atEOF; }
};
