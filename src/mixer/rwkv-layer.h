#ifndef RWKV_LAYER_H
#define RWKV_LAYER_H

#include <Eigen/Core>
#include <vector>
#include <math.h>

#include "lstm-layer.h"  // for NeuronLayer, Adam, AdamMatrix

// RWKV layer: replaces LSTM cell with WKV (Weighted Key Value) linear attention.
// 3 gates (same count as LSTM):
//   receptance (sigmoid) — gates output, like LSTM output gate
//   key (linear → exp)  — attention weight for current input
//   value (tanh)         — bounded input value, like LSTM input node
// State: (a, b, m) — numerator, denominator, log-space max stabilizer
// Output: sigmoid(r) * (a / b), where a/b is a weighted average of values.
class RwkvLayer {
 public:
  RwkvLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const Eigen::VectorXf& input, int input_symbol,
      Eigen::VectorXf* hidden, int hidden_start);
  void BackwardPass(const Eigen::VectorXf& input, int epoch,
      int layer, int input_symbol, Eigen::VectorXf* hidden_error,
      int bptt_start = 0);
  std::vector<Eigen::MatrixXf*> Weights();

 private:
  // WKV persistent state
  Eigen::VectorXf a_state_, b_state_, m_state_;

  // Per-epoch saved state for BPTT
  std::vector<Eigen::VectorXf> saved_a_, saved_b_, saved_m_;
  std::vector<Eigen::VectorXf> saved_exp_decay_, saved_exp_key_;
  std::vector<Eigen::VectorXf> saved_wkv_;

  // Error propagation
  Eigen::VectorXf stored_error_, a_error_, b_error_;

  // Learned decay parameter (per-channel, positive)
  Eigen::VectorXf w_, w_grad_, w_m_, w_v_;

  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  float beta1_power_ = 1.0f, beta2_power_ = 1.0f;

  NeuronLayer receptance_;  // sigmoid
  NeuronLayer key_;         // linear (into exp for attention weighting)
  NeuronLayer value_;       // tanh

  void ClipGradients(Eigen::VectorXf* arr);
  void ForwardPass(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int input_symbol);
  void BackwardPass(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int epoch, int layer, int input_symbol,
      Eigen::VectorXf* hidden_error, int bptt_start);
};
#include "rwkv-layer.hpp"

#endif
