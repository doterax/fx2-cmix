#include "gru.h"

#include <numeric>
#include <fstream>
#include <iostream>

inline Gru::Gru(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip, int bptt_depth, int bptt_period) : input_history_(horizon),
    hidden_(Eigen::VectorXf::Zero(num_cells * num_layers + 1)),
    hidden_error_(Eigen::VectorXf::Zero(num_cells)),
    layer_input_(horizon, std::vector<Eigen::VectorXf>(num_layers)),
    output_layer_(horizon, Eigen::MatrixXf::Zero(num_cells * num_layers + 1, output_size)),
    output_(horizon, Eigen::VectorXf::Constant(output_size, 1.0 / output_size)),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), input_size_(input_size), output_size_(output_size),
    bptt_depth_(bptt_depth <= 0 ? horizon : std::min(bptt_depth, horizon)),
    bptt_period_(bptt_period < 1 ? 1 : bptt_period) {
  hidden_[hidden_.size() - 1] = 1;

  for (int epoch = 0; epoch < horizon; ++epoch) {
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i] = Eigen::VectorXf::Zero(input_size + 1 + num_cells * 2);
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }

  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_,
        output_size_, num_cells, horizon, gradient_clip, learning_rate);
  }
}

inline Gru::~Gru() {}

inline void Gru::SetInput(const Eigen::VectorXf& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    layer_input_[epoch_][i].head(input_size_) = input.head(input_size_);
  }
}

inline Eigen::VectorXf& Gru::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0 && ++bptt_cycle_ >= bptt_period_) {
    bptt_cycle_ = 0;
    int bptt_start = horizon_ - bptt_depth_;
    for (int epoch = horizon_ - 1; epoch >= bptt_start; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        Eigen::VectorXf errors_vec = output_[epoch];
        errors_vec[input_history_[epoch]] -= 1.0f;
        hidden_error_.noalias() += output_layer_[epoch].block(
            offset, 0, num_cells_, output_size_) * errors_vec;
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_, bptt_start);
      }
    }
  }

  Eigen::VectorXf errors = output_[last_epoch];
  errors[input] -= 1.0f;
  errors *= learning_rate_;

  output_layer_[epoch_] = output_layer_[last_epoch];
  output_layer_[epoch_].noalias() -= hidden_ * errors.transpose();
  return Predict(input);
}

inline Eigen::VectorXf& Gru::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    layer_input_[epoch_][i].segment(input_size_, num_cells_) =
        hidden_.segment(i * num_cells_, num_cells_);
    layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_,
        i * num_cells_);
    if (i < layers_.size() - 1) {
      layer_input_[epoch_][i + 1].segment(num_cells_ + input_size_,
          num_cells_) = hidden_.segment(i * num_cells_, num_cells_);
    }
  }
  Eigen::VectorXf logits = output_layer_[epoch_].transpose() * hidden_;
  float max_logit = logits.maxCoeff();
  output_[epoch_] = (logits.array() - max_logit).exp();
  float total = output_[epoch_].sum();
  output_[epoch_] /= total;
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = input;
  return output_[epoch];
}
