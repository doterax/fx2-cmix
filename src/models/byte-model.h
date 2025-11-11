#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "model.h"

#include <Eigen/Core>
#include <vector>

class ByteModel : public Model {
 public:
  virtual ~ByteModel() {}
  ByteModel(const std::vector<bool>& vocab);
  const Eigen::VectorXf& BytePredict();
   Eigen::VectorXf& Predict() ;
  void Perceive(int bit);
  int ex;
 protected:
  void ByteUpdate();
  int top_, mid_, bot_;
  const std::vector<bool>& vocab_;
  Eigen::VectorXf probs_;
};

#endif

