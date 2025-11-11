#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>

class Model {
 public:
  Model() : outputs_(Eigen::VectorXf::Constant(1, 0.5)) {}
  Model(int size) : outputs_(Eigen::VectorXf::Constant(size, 0.5)) {}
  ~Model() {}
  const Eigen::VectorXf& Predict() const {return outputs_;}
  unsigned int NumOutputs() {return outputs_.size();}
  void Perceive(int bit) {}
  void ByteUpdate() {}

 protected:
  mutable Eigen::VectorXf outputs_;
};

#endif

