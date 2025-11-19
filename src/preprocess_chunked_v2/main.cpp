#include "bitstream.hpp"
#include "chunked_preprocessor_v2.hpp"
#include "text_utils.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// File container format:
// [varint control_size][varint ascii_size][varint
// utf8_size][control_bytes][ascii_bytes][utf8_bytes]

static void encode_varint_to(std::vector<uint8_t> &out, uint64_t v) {
  std::vector<uint8_t> tmp;
  encode_varint(v, tmp);
  out.insert(out.end(), tmp.begin(), tmp.end());
}
static bool decode_varint_mem(const std::vector<uint8_t> &buf, size_t &idx,
                              uint64_t &v) {
  v         = 0;
  int shift = 0;
  while (idx < buf.size()) {
    uint8_t b = buf[idx++];
    v |= uint64_t(b & 0x7F) << shift;
    if ((b & 0x80) == 0)
      return true;
    shift += 7;
    if (shift > 63)
      return false;
  }
  return false;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <c|d> <input> <output>" << std::endl;
    return 1;
  }
  std::string mode    = argv[1];
  std::string inPath  = argv[2];
  std::string outPath = argv[3];

  if (mode == "c") {
    // Load input file into memory
    std::ifstream fin(inPath, std::ios::binary);
    if (!fin) {
      std::cerr << "Failed to open input" << std::endl;
      return 1;
    }
    std::vector<uint8_t> inputBuf((std::istreambuf_iterator<char>(fin)),
                                  std::istreambuf_iterator<char>());
    fin.close();

    // Input bit reader
    class BitStreamReaderVector : public BitStreamReader {
      const std::vector<uint8_t> &buf;
      size_t                      idx = 0;
      int                         bit = 0;

    public:
      BitStreamReaderVector(const std::vector<uint8_t> &b) : buf(b) {}
      bool readBit() override {
        if (idx >= buf.size())
          return false;
        bool v = ((buf[idx] >> bit) & 1) != 0;
        if (++bit == 8) {
          bit = 0;
          ++idx;
        }
        return v;
      }
      uint64_t readBits(size_t n) override {
        uint64_t v = 0;
        for (size_t i = 0; i < n; ++i)
          v |= (uint64_t(readBit()) << i);
        return v;
      }
      bool eof() const override { return idx >= buf.size(); }
    } inputReader(inputBuf);

    // Output bit writers for control/ascii/utf8
    std::vector<uint8_t> controlBuf, asciiBuf, utf8Buf;
    class BitStreamWriterVector : public BitStreamWriter {
      std::vector<uint8_t> &buffer;
      uint8_t               cur = 0;
      int                   bp  = 0;

    public:
      BitStreamWriterVector(std::vector<uint8_t> &b) : buffer(b) {}
      void writeBit(bool bit) override {
        cur |= (bit ? 1 : 0) << bp;
        if (++bp == 8) {
          buffer.push_back(cur);
          cur = 0;
          bp  = 0;
        }
      }
      void writeBits(uint64_t v, size_t n) override {
        for (size_t i = 0; i < n; ++i)
          writeBit((v >> i) & 1);
      }
      void flush() override {
        if (bp > 0)
          buffer.push_back(cur);
        cur = 0;
        bp  = 0;
      }
    } cw(controlBuf), aw(asciiBuf), uw(utf8Buf);

    ChunkedV2Config cfg{};
    bool            ok =
        ChunkedPreprocessorV2::compress_stream(inputReader, cw, aw, uw, cfg);
    cw.flush();
    aw.flush();
    uw.flush();
    if (!ok) {
      std::cerr << "compress failed" << std::endl;
      return 1;
    }

    // Build container
    std::vector<uint8_t> outBuf;
    outBuf.reserve(controlBuf.size() + asciiBuf.size() + utf8Buf.size() + 32);
    encode_varint_to(outBuf, (uint64_t)controlBuf.size());
    encode_varint_to(outBuf, (uint64_t)asciiBuf.size());
    encode_varint_to(outBuf, (uint64_t)utf8Buf.size());
    outBuf.insert(outBuf.end(), controlBuf.begin(), controlBuf.end());
    outBuf.insert(outBuf.end(), asciiBuf.begin(), asciiBuf.end());
    outBuf.insert(outBuf.end(), utf8Buf.begin(), utf8Buf.end());

    std::ofstream fout(outPath, std::ios::binary);
    if (!fout) {
      std::cerr << "Failed to open output" << std::endl;
      return 1;
    }
    fout.write(reinterpret_cast<const char *>(outBuf.data()),
               (std::streamsize)outBuf.size());
    fout.close();
    return 0;
  } else if (mode == "d") {
    // Load container
    std::ifstream fin(inPath, std::ios::binary);
    if (!fin) {
      std::cerr << "Failed to open input" << std::endl;
      return 1;
    }
    std::vector<uint8_t> fileBuf((std::istreambuf_iterator<char>(fin)),
                                 std::istreambuf_iterator<char>());
    fin.close();
    size_t   idx = 0;
    uint64_t csz = 0, asz = 0, usz = 0;
    if (!decode_varint_mem(fileBuf, idx, csz) ||
        !decode_varint_mem(fileBuf, idx, asz) ||
        !decode_varint_mem(fileBuf, idx, usz)) {
      std::cerr << "Header decode failed" << std::endl;
      return 1;
    }
    if (idx + csz + asz + usz > fileBuf.size()) {
      std::cerr << "Size mismatch" << std::endl;
      return 1;
    }
    std::vector<uint8_t> controlBuf(fileBuf.begin() + idx,
                                    fileBuf.begin() + idx + csz);
    idx += csz;
    std::vector<uint8_t> asciiBuf(fileBuf.begin() + idx,
                                  fileBuf.begin() + idx + asz);
    idx += asz;
    std::vector<uint8_t> utf8Buf(fileBuf.begin() + idx,
                                 fileBuf.begin() + idx + usz);
    idx += usz;

    class BitStreamReaderVector : public BitStreamReader {
      const std::vector<uint8_t> &b;
      size_t                      pos = 0;
      int                         bit = 0;

    public:
      BitStreamReaderVector(const std::vector<uint8_t> &buf) : b(buf) {}
      bool readBit() override {
        if (pos >= b.size())
          return false;
        bool v = ((b[pos] >> bit) & 1) != 0;
        if (++bit == 8) {
          bit = 0;
          ++pos;
        }
        return v;
      }
      uint64_t readBits(size_t n) override {
        uint64_t v = 0;
        for (size_t i = 0; i < n; ++i)
          v |= (uint64_t(readBit()) << i);
        return v;
      }
      bool eof() const override { return pos >= b.size(); }
    } cr(controlBuf), ar(asciiBuf), ur(utf8Buf);

    // Output writer collects bytes then writes to file
    std::vector<uint8_t> outBytes;
    class BitStreamWriterVector : public BitStreamWriter {
      std::vector<uint8_t> &buf;
      uint8_t               cur = 0;
      int                   bp  = 0;

    public:
      BitStreamWriterVector(std::vector<uint8_t> &b) : buf(b) {}
      void writeBit(bool bit) override {
        cur |= (bit ? 1 : 0) << bp;
        if (++bp == 8) {
          buf.push_back(cur);
          cur = 0;
          bp  = 0;
        }
      }
      void writeBits(uint64_t v, size_t n) override {
        for (size_t i = 0; i < n; ++i)
          writeBit((v >> i) & 1);
      }
      void flush() override {
        if (bp > 0)
          buf.push_back(cur);
        cur = 0;
        bp  = 0;
      }
    } outWriter(outBytes);

    ChunkedV2Config cfg{};
    bool            ok =
        ChunkedPreprocessorV2::decompress_stream(cr, ar, ur, outWriter, cfg);
    outWriter.flush();
    if (!ok) {
      std::cerr << "decompress failed" << std::endl;
      return 1;
    }

    std::ofstream fout(outPath, std::ios::binary);
    if (!fout) {
      std::cerr << "Failed to open output" << std::endl;
      return 1;
    }
    fout.write(reinterpret_cast<const char *>(outBytes.data()),
               (std::streamsize)outBytes.size());
    fout.close();
    return 0;
  }

  std::cerr << "Unknown mode: " << mode << std::endl;
  return 1;
}
