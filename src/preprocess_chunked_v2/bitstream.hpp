#pragma once
#include <cstddef>
#include <cstdint>

class BitStreamWriter {
public:
  virtual ~BitStreamWriter() {}
  virtual void writeBit(bool bit)                      = 0;
  virtual void writeBits(uint64_t value, size_t count) = 0;
  virtual void flush()                                 = 0;
};

class BitStreamReader {
public:
  virtual ~BitStreamReader() {}
  virtual bool     readBit()              = 0;
  virtual uint64_t readBits(size_t count) = 0;
  virtual bool     eof() const            = 0;
};
