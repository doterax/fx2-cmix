#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stdio.h>
#include <string>

#include "../IPredictor.h"

namespace preprocessor {

typedef enum { DEFAULT, TEXT = 7 } Filetype;

void Encode(FILE *in, FILE *out, unsigned long long n,
            const std::string &temp_path, FILE *dictionary);

void NoPreprocess(FILE *in, FILE *out, unsigned long long n);

void Pretrain(IPredictor *p, FILE *dictionary);

void Decode(FILE *in, FILE *out, FILE *dictionary);

} // namespace preprocessor

#endif
