#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <string>

#include "lstm-layer.h"
#include "NMixer.h"

#include "../ds/SmallVector.h"

class Lstm: public NMixer {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  ~Lstm();
  Eigen::VectorXf& Perceive(unsigned int input) override;
  void SetInput(const Eigen::VectorXf& input) override;
private:
  Eigen::VectorXf& Predict(unsigned int input);

  //void SaveToDisk(const std::string& path);
  //void LoadFromDisk(const std::string& path);

 private:
  llvm::SmallVector<LstmLayer, 1> layers_;
  std::vector<uint8_t> input_history_; // horizon
  Eigen::VectorXf hidden_, hidden_error_;
  std::vector<std::vector<Eigen::VectorXf>> layer_input_;
  std::vector<Eigen::MatrixXf> output_layer_;  // [horizon] each (hidden_size+1) x output_size
  std::vector<Eigen::VectorXf> output_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  int last_input_ = -1;
};
#include "lstm.hpp"
#endif


