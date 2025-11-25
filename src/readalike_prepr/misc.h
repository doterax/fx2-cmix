
#define COMP_INTRO_END_LINE 29
#define COMP_MAIN_END_LINE  13146932
#define COMP_CODA_END_LINE  13147025

#define DECOMP_MAIN_END_LINE  13146905
#define DECOMP_INTRO_END_LINE 13146934 
#define DECOMP_CODA_END_LINE  13147027

void split4Comp(char const *enwik9_filename) {
  FILE* ifile = nullptr; fopen_s(&ifile, enwik9_filename, "rb");
  FILE* ofile1 = nullptr; fopen_s(&ofile1, ".intro", "wb");
  FILE* ofile2 = nullptr; fopen_s(&ofile2, ".main", "wb");
  FILE* ofile3 = nullptr; fopen_s(&ofile3, ".coda", "wb");  
  int line_count = 0;
  
  do {
    int c=getc(ifile);
    if (c==EOF) break;
    if (line_count < COMP_INTRO_END_LINE) {
      putc(c,ofile1);
    } else if (line_count < COMP_MAIN_END_LINE) {
      putc(c,ofile2);
    } else if (line_count < COMP_CODA_END_LINE) {
      putc(c,ofile3);
    } else {
      putc(c,ofile3);
    }
    if (c==10)
    line_count++;
  }
  while (!feof(ifile));
  fclose(ifile);
  fclose(ofile1);
  fclose(ofile2);
  fclose(ofile3);
}

void split4Decomp() {
  FILE* ifile = nullptr; fopen_s(&ifile, ".input_decomp", "rb");
  FILE* ofile1 = nullptr; fopen_s(&ofile1, ".intro_decomp", "wb");
  FILE* ofile2 = nullptr; fopen_s(&ofile2, ".main_decomp", "wb");
  FILE* ofile3 = nullptr; fopen_s(&ofile3, ".coda_decomp", "wb");  
  int line_count = 0;
  
  do {
      int c=getc(ifile);
      if (c==EOF) break;
    if (line_count < 13146906) {
      putc(c,ofile2);
    } else if (line_count < 13146935) {
        putc(c,ofile1);
    } else if (line_count < 13147027) {
        putc(c,ofile3);
    } else {
        putc(c,ofile3);
    }
    if (c==10)
    line_count++;
  }
  while (!feof(ifile));
  fclose(ifile);
  fclose(ofile1);
  fclose(ofile2);
  fclose(ofile3);
  
}
