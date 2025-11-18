#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

enum RegionType : uint8_t {
    REGION_RAW = 0,
    REGION_ASCII_SENTENCE = 1,
    REGION_BRACKETS = 2,
    REGION_WIKI_HEADER = 3,
    REGION_XML_TAG = 4,
    REGION_CURLY_BRACKETS = 5,
    REGION_UTF8_COMPRESSED = 6,
    REGION_RESERVED = 7
};

enum Transformation : uint8_t {
    TRANSFORM_NONE = 0,
    TRANSFORM_FIRST_UPPER = 1,
    TRANSFORM_ALL_UPPER = 2,
    TRANSFORM_RESERVED = 3
};

static const char* typeName(uint8_t t){
    switch(t){
        case REGION_RAW: return "RAW";
        case REGION_ASCII_SENTENCE: return "ASCII_SENTENCE";
        case REGION_BRACKETS: return "BRACKETS";
        case REGION_WIKI_HEADER: return "WIKI_HEADER";
        case REGION_XML_TAG: return "XML_TAG";
        case REGION_CURLY_BRACKETS: return "CURLY_BRACKETS";
        case REGION_UTF8_COMPRESSED: return "UTF8_COMPRESSED";
        default: return "RESERVED";
    }
}

static const char* transformName(uint8_t t){
    switch(t){
        case TRANSFORM_NONE: return "none";
        case TRANSFORM_FIRST_UPPER: return "first_upper";
        case TRANSFORM_ALL_UPPER: return "all_upper";
        default: return "reserved";
    }
}

int main(int argc, char** argv){
    if(argc != 2){
        std::fprintf(stderr, "Usage: %s <file.chunked>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    FILE* f = std::fopen(path, "rb");
    if(!f){
        std::perror("fopen");
        return 1;
    }
    size_t idx = 0;
    for(;;){
        int b = std::fgetc(f);
        if(b == EOF) break;
        uint8_t byte1 = (uint8_t)b;
        uint8_t type = byte1 & 0x0F;
        uint8_t transform = (byte1 >> 4) & 0x03;
        bool bigSize = (byte1 & 0x40) != 0;
        bool ctxDep  = (byte1 & 0x80) != 0;
        uint16_t size = 0;
        if(bigSize){
            int hi = std::fgetc(f); int lo = std::fgetc(f);
            if(hi==EOF||lo==EOF){ std::fprintf(stderr, "Truncated after header at chunk %zu\n", idx); break; }
            size = (uint16_t)((hi<<8)|lo);
        }else{
            int s = std::fgetc(f);
            if(s==EOF){ std::fprintf(stderr, "Missing size after header at chunk %zu\n", idx); break; }
            size = (uint16_t)s;
        }
        std::vector<uint8_t> data(size);
        size_t rd = std::fread(data.data(),1,size,f);
        if(rd != size){
            std::fprintf(stderr, "Truncated data at chunk %zu (expected %u, got %zu)\n", idx, size, rd);
            break;
        }
        // Print concise header and small preview
        std::printf("[%6zu] type=%s (%u) transform=%s big=%d ctx=%d size=%u ",
            idx, typeName(type), (unsigned)type, transformName(transform), bigSize?1:0, ctxDep?1:0, size);
        // Preview logic
        if(type == REGION_XML_TAG){
            if(!data.empty()){
                uint8_t taglen = data[0];
                std::string tag, content;
                if(1+taglen <= data.size()){
                    tag.assign((const char*)data.data()+1, (const char*)data.data()+1+taglen);
                    content.assign((const char*)data.data()+1+taglen, (const char*)data.data()+data.size());
                }
                if(content.size() > 24) content.resize(24);
                std::printf("tag=\"%s\" content=\"%s\"\n", tag.c_str(), content.c_str());
            } else {
                std::printf("tag=<empty>\n");
            }
        } else {
            // ASCII preview
            std::string s;
            for(size_t i=0;i<data.size() && i<32;i++){
                char c = (char)data[i];
                if(c<' '||c>126) c='.';
                s.push_back(c);
            }
            std::printf("data=\"%s\"\n", s.c_str());
        }
        idx++;
    }
    std::fclose(f);
    return 0;
}
