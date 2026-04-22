#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

class Sigmoid {
 public:
  Sigmoid(int logit_size);
  inline float Logit(float p) const {
    int index = static_cast<int>(p * logit_size_);
    if (index >= logit_size_) index = logit_size_ - 1;
    else if (index < 0) index = 0;
    return logit_table_[index];
  }
  static float Logistic(float p);
  static float FastLogistic(float p);

 private:
  float SlowLogit(float p);
  int logit_size_;
  std::vector<float> logit_table_;
};

#endif
