#ifndef XLSTM_H
#define XLSTM_H

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <string>

#include "xlstm-layer.h"
#include "NMixer.h"

#include "../ds/SmallVector.h"

class XLstm: public NMixer {
 public:
  XLstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip, int bptt_depth = 0, int bptt_period = 1);
  ~XLstm();
  Eigen::VectorXf& Perceive(unsigned int input) override;
  void SetInput(const Eigen::VectorXf& input) override;
private:
  Eigen::VectorXf& Predict(unsigned int input);

 private:
  llvm::SmallVector<XLstmLayer, 1> layers_;
  std::vector<uint8_t> input_history_;
  Eigen::VectorXf hidden_, hidden_error_;
  std::vector<std::vector<Eigen::VectorXf>> layer_input_;
  std::vector<Eigen::MatrixXf> output_layer_;
  std::vector<Eigen::VectorXf> output_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  int bptt_depth_;
  int bptt_period_;
  int bptt_cycle_ = 0;
  int last_input_ = -1;
};
#include "xlstm.hpp"
#endif
