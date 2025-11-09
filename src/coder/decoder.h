#ifndef DECODER_H
#define DECODER_H

#include <fstream>

#include "../IPredictor.h"

class Decoder {
public:
  Decoder(std::ifstream *is, IPredictor *p);
  int Decode();

private:
  int            ReadByte();
  unsigned int   Discretize(float p);

  std::ifstream *is_;
  unsigned int   x1_, x2_, x_;
  IPredictor    *p_;
};

#endif
