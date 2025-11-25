#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>

class Model {
protected:
  Model() : outputs_(Eigen::VectorXf::Constant(1, 0.5)) {}
  Model(int size) : outputs_(Eigen::VectorXf::Constant(size, 0.5)) {}

public:
  const Eigen::VectorXf &Predict() const { return outputs_; }
  unsigned int           NumOutputs() { return outputs_.size(); }

public:
  virtual ~Model()               = default;

  virtual void Perceive(int bit) = 0;
  virtual void ByteUpdate()      = 0;

protected:
  mutable Eigen::VectorXf outputs_;
};

#endif
