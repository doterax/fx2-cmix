#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "numbers_codec.hpp"

static bool read_varint_file(FILE *f, uint64_t &v) {
  v = 0; int shift = 0; while (true) { int b = fgetc(f); if (b == EOF) return false; v |= (uint64_t)(b & 0x7F) << shift; if ((b & 0x80) == 0) return true; shift += 7; if (shift > 63) return false; }
}

class BitStreamReaderVector {
  const std::vector<uint8_t> &buf; size_t bp=0; int bit=0;
public:
  BitStreamReaderVector(const std::vector<uint8_t>&b):buf(b){}
  bool eof() const { return bp >= buf.size(); }
  uint64_t readBits(size_t n){ uint64_t v=0; for(size_t i=0;i<n;++i){ if(bp>=buf.size()) return v; v |= ((uint64_t)((buf[bp]>>bit)&1))<<i; if(++bit==8){ bit=0; ++bp; } } return v; }
};

static const char* typeName(uint8_t t){ switch(t){ case 0: return "RAW"; case 1: return "NUMBER"; case 2: return "BRACKETS"; case 3: return "WIKI_HEADER"; case 4: return "XML_TAG"; case 5: return "CURLY_BRACKETS"; case 6: return "UTF8_RUN"; case 7: return "RESERVED"; default: return "UNKNOWN"; } }
static const char* transformName(uint8_t t){ switch(t){ case 0: return "none"; case 1: return "first_upper"; case 2: return "all_upper"; case 3: return "reserved"; default: return "?"; } }

int main(int argc, char **argv){ if(argc<2){ std::cerr << "Usage: " << argv[0] << " <container>" << std::endl; return 1; }
  std::string containerPath = argv[1]; FILE *f = fopen(containerPath.c_str(), "rb"); if(!f){ std::cerr << "open failed" << std::endl; return 1; }
  uint64_t csz=0, asz=0, usz=0, nsz=0, ncount=0; if(!read_varint_file(f,csz)||!read_varint_file(f,asz)||!read_varint_file(f,usz)||!read_varint_file(f,nsz)||!read_varint_file(f,ncount)){ fclose(f); std::cerr << "header failed" << std::endl; return 1; }
  std::vector<uint8_t> control(csz), ascii(asz), utf8(usz), numbersBytes(nsz); if(csz && fread(control.data(),1,csz,f)!=csz){ fclose(f); std::cerr << "control read fail"<<std::endl; return 1;} if(asz && fread(ascii.data(),1,asz,f)!=asz){ fclose(f); std::cerr << "ascii read fail"<<std::endl; return 1;} if(usz && fread(utf8.data(),1,usz,f)!=usz){ fclose(f); std::cerr << "utf8 read fail"<<std::endl; return 1;} if(nsz && fread(numbersBytes.data(),1,nsz,f)!=nsz){ fclose(f); std::cerr << "numbers read fail"<<std::endl; return 1;} fclose(f);
  std::vector<uint64_t> numbers; if(!decode_numbers_gamma(numbersBytes,ncount,numbers) || numbers.size()!=ncount){ std::cerr << "numbers gamma decode failed"<<std::endl; return 1; }
  std::cout << "control="<<csz<<" ascii="<<asz<<" utf8="<<usz<<" numbers_bytes="<<nsz<<" numbers_count="<<numbers.size()<<"\n";
  BitStreamReaderVector cr(control); size_t ascii_pos=0, utf8_pos=0, number_index=0, idx=0; while(number_index < numbers.size()){
    uint64_t type = cr.readBits(3); if(cr.eof()) break; uint64_t transform = cr.readBits(2); if(number_index>=numbers.size()){ std::cerr << "numbers underrun"<<std::endl; break; }
    uint64_t len=0; uint64_t numberVal=0; uint64_t numDigits=0;
    if(type==1){ // NUMBER uses two entries: run_len then value
      if(number_index + 1 >= numbers.size()){ std::cerr << "numbers underrun for NUMBER" << std::endl; break; }
      numDigits = numbers[number_index++];
      numberVal = numbers[number_index++];
      // reconstruct with zero-padding
      std::string digits = (numberVal==0?"0":"");
      if(numberVal>0){ std::string tmp; uint64_t v=numberVal; while(v>0){ tmp.push_back(char('0'+(v%10))); v/=10; } std::reverse(tmp.begin(), tmp.end()); digits.swap(tmp); }
      if(digits.size()<numDigits) digits = std::string(numDigits - digits.size(),'0') + digits;
      len = 0; // no ascii payload stored
      std::string preview = digits;
      uint64_t headerBits=5; uint64_t headerBytes=(headerBits+7)/8;
      std::printf("[%6zu] type=%s (%u) transform=%s digits=%s (len=%llu) hdr=%llu\n", idx,typeName((uint8_t)type),(unsigned)type,transformName((uint8_t)transform), preview.c_str(), (unsigned long long)numDigits,(unsigned long long)headerBytes);
      idx++; continue;
    } else { len = numbers[number_index++]; }
    std::string preview;
    if(type==6){ if(utf8_pos+len<=utf8.size()){ for(size_t i=0;i<len && i<32;i++){ uint8_t c=utf8[utf8_pos+i]; if(c<32) c='.'; preview.push_back((char)c);} } }
    else { if(ascii_pos+len<=ascii.size()){ if(type==4){ const uint8_t* base=&ascii[ascii_pos]; size_t sep=0; while(sep<len && base[sep]!=0) sep++; std::string tag, content; for(size_t i=0;i<sep && i<16;i++){ char ch=(char)base[i]; if(ch<' '||ch>126) ch='.'; tag.push_back(ch);} size_t contentStart=(sep<len)?sep+1:len; for(size_t i=contentStart;i<len && (i-contentStart)<24;i++){ char ch=(char)base[i]; if(ch<' '||ch>126) ch='.'; content.push_back(ch);} preview = "tag=\""+tag+"\" content=\""+content+"\""; }
      else { for(size_t i=0;i<len && i<32;i++){ uint8_t c=ascii[ascii_pos+i]; if(c<' '||c>126) c='.'; preview.push_back((char)c);} } } }
    uint64_t headerBits=5; uint64_t headerBytes=(headerBits+7)/8; std::printf("[%6zu] type=%s (%u) transform=%s len=%llu hdr=%llu data=", idx,typeName((uint8_t)type),(unsigned)type,transformName((uint8_t)transform),(unsigned long long)len,(unsigned long long)headerBytes); if(!preview.empty()) std::printf("%s\n", preview.c_str()); else std::printf("<no-data>\n");
    if(type==6) utf8_pos += (size_t)len; else if(type!=1) ascii_pos += (size_t)len; idx++; if(ascii_pos>ascii.size() || utf8_pos>utf8.size()){ std::fprintf(stderr,"Stream overrun at chunk %zu (pos exceeded)\n", idx); break; }
  }
  std::cout << "\nNumbers: gamma_bytes=" << nsz << " count=" << numbers.size() << "\n";
  return 0; }
