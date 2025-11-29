#pragma once

#include "NMixer.h"
#include <Eigen/Core>

class NullMixer : public NMixer {
private:
  Eigen::VectorXf tmp_;
public:
  NullMixer(int input_size);
  Eigen::VectorXf &Perceive(unsigned int input) override;
  Eigen::VectorXf &Predict(unsigned int input) override;
  void             SetInput(const Eigen::VectorXf &input) override;
  ~NullMixer();
};

NullMixer::NullMixer(int input_size) : tmp_(input_size) {
    tmp_.setZero();
}

Eigen::VectorXf &NullMixer::Perceive(unsigned int input) {
    return Predict(input);
}

Eigen::VectorXf &NullMixer::Predict(unsigned int input) {
    return tmp_;
}

void NullMixer::SetInput(const Eigen::VectorXf &input) {
    tmp_ = input;
}

NullMixer::~NullMixer() {}
