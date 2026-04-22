#ifndef MIXER_INPUT_H
#define MIXER_INPUT_H

#include "sigmoid.h"

#include <Eigen/Core>
#include <algorithm>
#include <vector>

class MixerInput {
 public:
  MixerInput(const Sigmoid& sigmoid, float eps);
  void SetNumModels(int num_models);
  inline void SetInput(int index, float p) {
    p = std::min(std::max(p, min_), max_);
    inputs_[index] = sigmoid_.Logit(p);
  }
  inline void SetStretchedInput(int index, float p) {
    inputs_[index] = std::min(std::max(p, stretched_min_), stretched_max_);
  }
  inline void SetZero(int index) { inputs_[index] = 0.0f; }
  inline void SetExtraInput(size_t index, float p) {
    extra_inputs_[index] = std::min(std::max(p, stretched_min_), stretched_max_);
  }
  void SetExtraInputSize(size_t size) { extra_inputs_.resize(size);};
  //void ClearExtraInputs() { extra_inputs_.clear(); }
  const Eigen::VectorXf& Inputs() const { return inputs_; }
  //const std::vector<float>& ExtraInputs() const { return extra_inputs_; }
  const auto& ExtraInputs() const { return extra_inputs_; }

 private:
  Eigen::VectorXf inputs_;
  //std::vector<float> extra_inputs_;
  Eigen::VectorXf extra_inputs_;
  const Sigmoid& sigmoid_;
  float min_, max_, stretched_min_, stretched_max_;
};

#endif

