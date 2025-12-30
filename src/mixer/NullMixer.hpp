#pragma once

#include "NMixer.h"
#include <Eigen/Core>

class NullMixer : public NMixer {
private:
  Eigen::VectorXf tmp_;
public:
  NullMixer(int input_size);
  Eigen::VectorXf &Perceive(unsigned int input) override;
  void             SetInput(const Eigen::VectorXf &input) override;
  ~NullMixer();
};

NullMixer::NullMixer(int input_size) : tmp_(input_size) {
    tmp_.setZero();
}

Eigen::VectorXf &NullMixer::Perceive(unsigned int input) {
    return tmp_;
}

void NullMixer::SetInput(const Eigen::VectorXf &input) {
   // check input size
   assert(input.size() == tmp_.size());

   //check probabilities in range 0..1
   for(int i = 0; i != input.size(); ++i) {
        assert(input[i] >= 0.0f && input[i] <= 1.0f);
    }

    //check sum to 1.0
    float sum = input.sum();
    assert(sum > 0.999f && sum < 1.001f);

    tmp_ = input;
}

NullMixer::~NullMixer() {}
