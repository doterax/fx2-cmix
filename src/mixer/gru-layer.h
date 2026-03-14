#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "lstm-layer.h" // for NeuronLayer, Adam, AdamMatrix

class GruLayer {
 public:
  GruLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const Eigen::VectorXf& input, int input_symbol,
      Eigen::VectorXf* hidden, int hidden_start);
  void BackwardPass(const Eigen::VectorXf& input, int epoch,
      int layer, int input_symbol, Eigen::VectorXf* hidden_error,
      int bptt_start = 0);
  std::vector<Eigen::MatrixXf*> Weights();

 private:
  Eigen::VectorXf stored_error_;
  std::vector<Eigen::VectorXf> last_hidden_;   // h_{t-1} per epoch
  std::vector<Eigen::VectorXf> reset_hidden_;  // r_t * h_{t-1} per epoch
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  float beta1_power_ = 1.0f, beta2_power_ = 1.0f;
  NeuronLayer reset_gate_, update_gate_, candidate_;

  void ClipGradients(Eigen::VectorXf* arr);
  void ForwardPassGate(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int input_symbol);
  void BackwardPassGate(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int epoch, int layer, int input_symbol,
      Eigen::VectorXf* hidden_error, int bptt_start);
};
#include "gru-layer.hpp"

#endif
