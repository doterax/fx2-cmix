#ifndef SELF_EXTRACT_H 
#define SELF_EXTRACT_H 

#include <malloc.h>

#include <string>

struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int decomp_input_size;
};

void write(const std::string& file_name, HeaderInfo& data) {
  FILE *out = nullptr;
  if (fopen_s(&out, file_name.c_str() , "wb") != 0 || !out) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", file_name.c_str());
    abort();
  }
  fwrite(&data , 1 , sizeof(HeaderInfo) , out );
  fclose(out);
}

void read(const std::string& file_name, HeaderInfo& data) {
  FILE *in = nullptr;
  if (fopen_s(&in, file_name.c_str() , "rb") != 0 || !in) {
    fprintf(stderr, "Error: Cannot open file for reading: %s\n", file_name.c_str());
    abort();
  }
  fread(&data , 1 , sizeof(HeaderInfo) , in );
  fclose(in);
}


// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it) 
int selfextract_comp() {
  HeaderInfo header;

// open itslef to read auxilary data (dictionary and neworder)
  FILE *f = nullptr, *fo = nullptr;
  if (fopen_s(&f, "cmix", "rb") != 0 || !f) {
    fprintf(stderr, "Error: Cannot open file for reading: %s\n", "cmix");
    abort();
  }

  // get the size of the whole binary
  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  // read the whole binary to memory
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = nullptr;
  if (fopen_s(&fo, "test.dat", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", "test.dat");
    abort();
  }
  memcpy(&header, p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo));
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);

  //Remove dictionary if present
  remove(".dict");
  
  size_t decmpressor_binary_size = fsize - header.dict_size - header.new_article_order_size - sizeof(HeaderInfo);

// produce actual decompressor binary 
  fo = nullptr;
  if (fopen_s(&fo, ".decomp_bin", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", ".decomp_bin");
    abort();
  }
  fwrite(p1, decmpressor_binary_size, 1, fo);
  fclose(fo);

// produce dictionary and decompress it
  fo = nullptr;
  if (fopen_s(&fo, ".dict.comp", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", ".dict.comp");
    abort();
  }
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);


// produce article order and decompress it
  fo = nullptr;
  if (fopen_s(&fo, ".new_article_order.comp", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", ".new_article_order.comp");
    abort();
  }
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.new_article_order_size, 1, fo);
  fclose(fo);
//  std::cout << "Decompressing the file with the new article order..." << std::endl;
  system("./cmix -d .new_article_order.comp .new_article_order");

//  std::cout << "Decompressing dictionary..." << std::endl;
  system("./cmix -d .dict.comp .dict");
  free(p1);
  return 0;
}

// Same as previous function, but used in decompressor
// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it) 
int selfextract_decomp() {
  HeaderInfo header;
  FILE *f = nullptr, *fo = nullptr;
  if (fopen_s(&f, "archive9", "rb") != 0 || !f) {
    fprintf(stderr, "Error: Cannot open file for reading: %s\n", "archive9");
    abort();
  }

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = nullptr;
  if (fopen_s(&fo, "test.dat", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", "test.dat");
    abort();
  }
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);
  read("test.dat", header);

  //Remove dictionary if present
  remove(".dict");
  
  size_t decmpressor_binary_size = fsize - header.dict_size - header.decomp_input_size - sizeof(HeaderInfo);

  fo = nullptr;
  if (fopen_s(&fo, ".dict.comp_decomp", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", ".dict.comp_decomp");
    abort();
  }
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);

  system("./archive9 -d .dict.comp_decomp .dict");//_decomp

  fo = nullptr;
  if (fopen_s(&fo, ".ready4cmix_decomp", "wb") != 0 || !fo) {
    fprintf(stderr, "Error: Cannot open file for writing: %s\n", ".ready4cmix_decomp");
    abort();
  }
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.decomp_input_size, 1, fo);
  fclose(fo);

  free(p1);
  return 0;
}

#endif // PREPR_H
