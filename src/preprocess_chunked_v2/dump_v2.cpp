#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

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

static const char *typeName(uint8_t t) {
  switch (t) {
  case 0:
    return "RAW";
  case 1:
    return "ASCII_SENTENCE";
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
}
static const char *transformName(uint8_t t) {
  switch (t) {
  case 0:
    return "none";
  case 1:
    return "first_upper";
  case 2:
    return "all_upper";
  case 3:
    return "reserved";
  default:
    return "?";
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <container>" << std::endl;
    return 1;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    std::cerr << "open failed" << std::endl;
    return 1;
  }
  uint64_t csz = 0, asz = 0, usz = 0;
  if (!read_varint_file(f, csz) || !read_varint_file(f, asz) ||
      !read_varint_file(f, usz)) {
    fclose(f);
    std::cerr << "header failed" << std::endl;
    return 1;
  }
  std::vector<uint8_t> control(csz), ascii(asz), utf8(usz);
  fread(control.data(), 1, csz, f);
  fread(ascii.data(), 1, asz, f);
  fread(utf8.data(), 1, usz, f);
  fclose(f);
  std::cout << "control=" << csz << " ascii=" << asz << " utf8=" << usz << "\n";
  BitStreamReaderVector cr(control);
  // Maintain cursors into ascii and utf8 byte streams
  size_t ascii_pos = 0, utf8_pos = 0;
  size_t idx = 0;
  while (!cr.eof()) {
    uint64_t bitsBefore = cr.consumedBits();
    uint64_t type       = cr.readBits(3);
    uint64_t transform  = cr.readBits(2);
    uint64_t len        = 0;
    int      shift      = 0;
    while (true) {
      uint64_t b = cr.readBits(8);
      len |= (b & 0x7F) << shift;
      if ((b & 0x80) == 0)
        break;
      shift += 7;
    }
    uint64_t bitsAfter   = cr.consumedBits();
    uint64_t headerBits  = bitsAfter - bitsBefore;
    uint64_t headerBytes = (headerBits + 7) / 8;
    if (type > 7)
      type = 7;
    if (transform > 3)
      transform = 3;
    // Fetch preview data without reconstructing wrappers; show stored payload
    std::string preview;
    if (type == 6) { // UTF8_RUN
      if (utf8_pos + len <= utf8.size()) {
        for (size_t i = 0; i < len && i < 32; i++) {
          uint8_t c = utf8[utf8_pos + i];
          // leave UTF-8 bytes; map control chars to '.'
          if (c < 32)
            c = '.';
          preview.push_back((char)c);
        }
      }
    } else {
      if (ascii_pos + len <= ascii.size()) {
        if (type == 4) { // XML_TAG stored: tag + 0 + content
          // find separator for preview
          const uint8_t *base = &ascii[ascii_pos];
          size_t         sep  = 0;
          while (sep < len && base[sep] != 0)
            sep++;
          std::string tag, content;
          for (size_t i = 0; i < sep && i < 16; i++) {
            char ch = (char)base[i];
            if (ch < ' ' || ch > 126)
              ch = '.';
            tag.push_back(ch);
          }
          size_t contentStart = (sep < len) ? sep + 1 : len;
          for (size_t i = contentStart; i < len && (i - contentStart) < 24;
               ++i) {
            char ch = (char)base[i];
            if (ch < ' ' || ch > 126)
              ch = '.';
            content.push_back(ch);
          }
          preview = "tag=\"" + tag + "\" content=\"" + content + "\"";
        } else {
          for (size_t i = 0; i < len && i < 32; i++) {
            uint8_t c = ascii[ascii_pos + i];
            if (c < ' ' || c > 126)
              c = '.';
            preview.push_back((char)c);
          }
          if (type == 1 && len < 32)
            preview.push_back(
                '.'); // sentence ends with period reconstructed later
        }
      }
    }
    std::printf("[%6zu] type=%s (%u) transform=%s len=%llu hdr=%llu data=", idx,
                typeName((uint8_t)type), (unsigned)type,
                transformName((uint8_t)transform), (unsigned long long)len,
                (unsigned long long)headerBytes);
    if (!preview.empty())
      std::printf("%s\n", preview.c_str());
    else
      std::printf("<no-data>\n");
    // Advance stream cursors
    if (type == 6) {
      utf8_pos += (size_t)len;
    } else {
      ascii_pos += (size_t)len;
    }
    idx++;
    // Safety break if payload pointers exceed available to avoid infinite loop
    // on malformed input
    if (ascii_pos > ascii.size() || utf8_pos > utf8.size()) {
      std::fprintf(stderr, "Stream overrun at chunk %zu\n", idx);
      break;
    }
  }
  return 0;
}
