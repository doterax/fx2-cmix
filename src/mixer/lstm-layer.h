#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include <Eigen/Core>
#include <vector>
#include <math.h>

struct NeuronLayer {
  NeuronLayer(unsigned int input_size, unsigned int num_cells, int horizon,
    int offset) : error_(Eigen::VectorXf::Zero(num_cells)), 
    ivar_(Eigen::VectorXf::Zero(horizon)), 
    gamma_(Eigen::VectorXf::Constant(num_cells, 1.0)),
    gamma_u_(Eigen::VectorXf::Zero(num_cells)), 
    gamma_m_(Eigen::VectorXf::Zero(num_cells)), 
    gamma_v_(Eigen::VectorXf::Zero(num_cells)),
    beta_(Eigen::VectorXf::Zero(num_cells)), 
    beta_u_(Eigen::VectorXf::Zero(num_cells)), 
    beta_m_(Eigen::VectorXf::Zero(num_cells)),
    beta_v_(Eigen::VectorXf::Zero(num_cells)),
    weights_(Eigen::MatrixXf::Zero(num_cells, input_size)),
    update_(Eigen::MatrixXf::Zero(num_cells, input_size)),
    m_(Eigen::MatrixXf::Zero(num_cells, input_size)),
    v_(Eigen::MatrixXf::Zero(num_cells, input_size)),
    recurrent_weights_(Eigen::MatrixXf::Zero(num_cells, input_size - offset)),
    state_(horizon, Eigen::VectorXf::Zero(num_cells)),
    norm_(horizon, Eigen::VectorXf::Zero(num_cells)) {};

  Eigen::VectorXf error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  Eigen::MatrixXf weights_, update_, m_, v_, recurrent_weights_;
  std::vector<Eigen::VectorXf> state_, norm_;
};

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const Eigen::VectorXf& input, int input_symbol,
      Eigen::VectorXf* hidden, int hidden_start);
  void BackwardPass(const Eigen::VectorXf& input, int epoch,
      int layer, int input_symbol, Eigen::VectorXf* hidden_error,
      int bptt_start = 0);
  std::vector<Eigen::MatrixXf*> Weights();

 private:
  Eigen::VectorXf state_, state_error_, stored_error_;
  std::vector<Eigen::VectorXf> tanh_state_, input_gate_state_,
      last_state_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0;
  float beta1_power_ = 1.0f, beta2_power_ = 1.0f;
  NeuronLayer forget_gate_, input_node_, output_gate_;

  void ClipGradients(Eigen::VectorXf* arr);
  void ForwardPass(NeuronLayer& neurons, const Eigen::VectorXf& input,
      int input_symbol);
  void BackwardPass(NeuronLayer& neurons, const Eigen::VectorXf&input,
      int epoch, int layer, int input_symbol,
      Eigen::VectorXf* hidden_error, int bptt_start);
};
#include "lstm-layer.hpp"

#endif

