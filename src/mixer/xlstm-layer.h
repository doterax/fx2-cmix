#ifndef XLSTM_LAYER_H
#define XLSTM_LAYER_H

#include "lstm-layer.h" // for NeuronLayer, Adam, AdamMatrix

// sLSTM layer: exponential gating + normalizer state.
// 4 gates (forget, input, output, candidate) vs LSTM's coupled 3.
class XLstmLayer {
 public:
  XLstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const Eigen::VectorXf& input, int input_symbol,
      Eigen::VectorXf* hidden, int hidden_start);
  void BackwardPass(const Eigen::VectorXf& input, int epoch,
      int layer, int input_symbol, Eigen::VectorXf* hidden_error,
      int bptt_start = 0);
  std::vector<Eigen::MatrixXf*> Weights();

 private:
  // Running state
  Eigen::VectorXf cell_state_, normalizer_, m_state_;
  // Gradient accumulators
  Eigen::VectorXf cell_error_, norm_error_, stored_error_;
  // Per-timestep storage for BPTT
  std::vector<Eigen::VectorXf> prev_cell_, prev_norm_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  float beta1_power_ = 1.0f, beta2_power_ = 1.0f;
  NeuronLayer forget_gate_, input_gate_, output_gate_, candidate_;

  void ClipGradients(Eigen::VectorXf* arr);
  void ForwardPassGate(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int input_symbol);
  void BackwardPassGate(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int epoch, int layer, int input_symbol,
      Eigen::VectorXf* hidden_error, int bptt_start);
};
#include "xlstm-layer.hpp"

#endif
