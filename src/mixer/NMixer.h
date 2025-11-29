#pragma once
#include <Eigen/Core>

class NMixer {
public:
  virtual Eigen::VectorXf &Perceive(unsigned int input)           = 0;
  virtual Eigen::VectorXf &Predict(unsigned int input)            = 0;
  virtual void             SetInput(const Eigen::VectorXf &input) = 0;
  virtual ~NMixer()                                               = default;
};
