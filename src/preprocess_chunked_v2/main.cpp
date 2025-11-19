#include "bitstream.hpp"
#include "chunked_preprocessor_v2.hpp"
#include "text_utils.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "numbers_codec.hpp"

// File container format (gamma numbers embedded):
// [varint control_size][varint ascii_size][varint utf8_size][varint numbers_bytes][varint numbers_count]
// [control_bytes][ascii_bytes][utf8_bytes][numbers_gamma_bitstream]

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

// Extract embedded streams from a written container file and write them to
// sibling files: <path>.control, <path>.ascii, <path>.utf8, <path>.numbers.
// Returns true on success.
static bool extract_embedded_streams(const std::string &containerPath) {
  std::ifstream fin(containerPath, std::ios::binary);
  if (!fin) {
    std::cerr << "[extract] open failed: " << containerPath << "\n";
    return false;
  }
  std::vector<uint8_t> fileBuf((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  fin.close();
  size_t idx = 0; uint64_t csz=0, asz=0, usz=0, nsz=0, ncount=0;
  if (!decode_varint_mem(fileBuf, idx, csz) || !decode_varint_mem(fileBuf, idx, asz) ||
      !decode_varint_mem(fileBuf, idx, usz) || !decode_varint_mem(fileBuf, idx, nsz) ||
      !decode_varint_mem(fileBuf, idx, ncount)) {
    std::cerr << "[extract] header decode failed\n"; return false; }
  if (idx + csz + asz + usz + nsz > fileBuf.size()) { std::cerr << "[extract] size mismatch\n"; return false; }
  const uint8_t *controlPtr = fileBuf.data() + idx; idx += csz;
  const uint8_t *asciiPtr   = fileBuf.data() + idx; idx += asz;
  const uint8_t *utf8Ptr    = fileBuf.data() + idx; idx += usz;
  const uint8_t *numbersPtr = fileBuf.data() + idx; /* idx += nsz; */
  auto writeFile = [](const std::string &p, const uint8_t *data, size_t sz) {
    std::ofstream f(p, std::ios::binary); if (!f) { std::cerr << "[extract] cannot write " << p << "\n"; return false; }
    if (sz) f.write(reinterpret_cast<const char*>(data), (std::streamsize)sz); return true; };
  bool ok = true;
  ok &= writeFile(containerPath + ".control", controlPtr, (size_t)csz);
  ok &= writeFile(containerPath + ".ascii",   asciiPtr,   (size_t)asz);
  ok &= writeFile(containerPath + ".utf8",    utf8Ptr,    (size_t)usz);
  ok &= writeFile(containerPath + ".numbers", numbersPtr, (size_t)nsz);
  if (!ok) return false;
  std::cout << "Extracted streams: control=" << csz << " ascii=" << asz << " utf8=" << usz << " numbers_bytes=" << nsz << " count=" << ncount << "\n";
  return true;
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

    ChunkedV2Config       cfg{};
    cfg.min_raw_context_before_number_split = 0; // disable gating to ensure correct NUMBER boundaries
    std::vector<uint64_t> numbers; // lengths + NUMBER values
    bool ok = ChunkedPreprocessorV2::compress_stream(inputReader, cw, aw, uw, cfg, &numbers);
    std::cout << "Compressed " << inputBuf.size() << " bytes into "
              << controlBuf.size() << " control bytes, " << asciiBuf.size()
              << " ascii bytes, " << utf8Buf.size() << " utf8 bytes."
              << std::endl;
    cw.flush();
    aw.flush();
    uw.flush();
    if (!ok) {
      std::cerr << "compress failed" << std::endl;
      return 1;
    }

    // Build container (extended 4-stream with embedded gamma numbers)
    auto numbersBytes = encode_numbers_gamma(numbers);
    std::vector<uint8_t> outBuf;
    outBuf.reserve(controlBuf.size() + asciiBuf.size() + utf8Buf.size() + numbersBytes.size() + 48);
    encode_varint_to(outBuf, (uint64_t)controlBuf.size());
    encode_varint_to(outBuf, (uint64_t)asciiBuf.size());
    encode_varint_to(outBuf, (uint64_t)utf8Buf.size());
    encode_varint_to(outBuf, (uint64_t)numbersBytes.size());
    encode_varint_to(outBuf, (uint64_t)numbers.size());
    outBuf.insert(outBuf.end(), controlBuf.begin(), controlBuf.end());
    outBuf.insert(outBuf.end(), asciiBuf.begin(), asciiBuf.end());
    outBuf.insert(outBuf.end(), utf8Buf.begin(), utf8Buf.end());
    outBuf.insert(outBuf.end(), numbersBytes.begin(), numbersBytes.end());

    std::ofstream fout(outPath, std::ios::binary);
    if (!fout) {
      std::cerr << "Failed to open output" << std::endl;
      return 1;
    }
    fout.write(reinterpret_cast<const char *>(outBuf.data()),
               (std::streamsize)outBuf.size());
    fout.close();
    std::cout << "Numbers embedded: count=" << numbers.size() << " gamma_bytes=" << numbersBytes.size() << "\n";
    // Re-open written container to reconstruct streams
    if (!extract_embedded_streams(outPath)) {
      std::cerr << "Failed to extract substreams" << std::endl;
      return 1;
    }
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
    uint64_t csz = 0, asz = 0, usz = 0, nsz = 0, ncount = 0;
    if (!decode_varint_mem(fileBuf, idx, csz) ||
        !decode_varint_mem(fileBuf, idx, asz) ||
        !decode_varint_mem(fileBuf, idx, usz) ||
        !decode_varint_mem(fileBuf, idx, nsz) ||
        !decode_varint_mem(fileBuf, idx, ncount)) {
      std::cerr << "Header decode failed" << std::endl;
      return 1;
    }
    if (idx + csz + asz + usz + nsz > fileBuf.size()) {
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
    std::vector<uint8_t> numbersBytes(fileBuf.begin() + idx,
                      fileBuf.begin() + idx + nsz);
    idx += nsz;

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
    std::vector<uint64_t> numbers;
    if (!decode_numbers_gamma(numbersBytes, ncount, numbers) || numbers.size() != ncount) {
      std::cerr << "Failed to decode embedded numbers" << std::endl;
      return 1;
    }
    bool ok = ChunkedPreprocessorV2::decompress_stream(cr, ar, ur, outWriter, cfg, numbers.empty()?nullptr:&numbers);
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
    std::cout << "Numbers decoded: count=" << numbers.size() << "\n";
    return 0;
  }

  std::cerr << "Unknown mode: " << mode << std::endl;
  return 1;
}
