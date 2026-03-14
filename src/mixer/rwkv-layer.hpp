#include "rwkv-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>
#include "..\random.hpp"

// Adam() and AdamMatrix() are already defined in lstm-layer.hpp
// (included via lstm-layer.h -> rwkv-layer.h)

inline RwkvLayer::RwkvLayer(unsigned int input_size,
    unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    a_state_(Eigen::VectorXf::Zero(num_cells)),
    b_state_(Eigen::VectorXf::Zero(num_cells)),
    m_state_(Eigen::VectorXf::Constant(num_cells, -1e4f)),
    saved_a_(horizon, Eigen::VectorXf::Zero(num_cells)),
    saved_b_(horizon, Eigen::VectorXf::Zero(num_cells)),
    saved_m_(horizon, Eigen::VectorXf::Zero(num_cells)),
    saved_exp_decay_(horizon, Eigen::VectorXf::Zero(num_cells)),
    saved_exp_key_(horizon, Eigen::VectorXf::Zero(num_cells)),
    saved_wkv_(horizon, Eigen::VectorXf::Zero(num_cells)),
    stored_error_(Eigen::VectorXf::Zero(num_cells)),
    a_error_(Eigen::VectorXf::Zero(num_cells)),
    b_error_(Eigen::VectorXf::Zero(num_cells)),
    w_(Eigen::VectorXf::Constant(num_cells, 0.5f)),
    w_grad_(Eigen::VectorXf::Zero(num_cells)),
    w_m_(Eigen::VectorXf::Zero(num_cells)),
    w_v_(Eigen::VectorXf::Zero(num_cells)),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    receptance_(input_size, num_cells, horizon, output_size_ + input_size_),
    key_(input_size, num_cells, horizon, output_size_ + input_size_),
    value_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;

  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < receptance_.weights_.cols(); ++j) {
      receptance_.weights_(i, j) = low + get_uniform_random() * range;
      key_.weights_(i, j) = low + get_uniform_random() * range;
      value_.weights_(i, j) = low + get_uniform_random() * range;
    }
  }
}

inline void RwkvLayer::ForwardPass(const Eigen::VectorXf& input,
    int input_symbol, Eigen::VectorXf* hidden, int hidden_start) {
  // Save pre-update WKV state for BPTT
  saved_a_[epoch_] = a_state_;
  saved_b_[epoch_] = b_state_;
  saved_m_[epoch_] = m_state_;

  // Linear transforms + layer norm
  ForwardPass(receptance_, input, input_symbol);
  ForwardPass(key_, input, input_symbol);
  ForwardPass(value_, input, input_symbol);

  // Activations: sigmoid(r), identity(k), tanh(v)
  receptance_.state_[epoch_] = (1.0f / (1.0f +
      (-receptance_.state_[epoch_].array()).exp())).matrix();
  // key: no activation (linear, goes into exp for attention weighting)
  value_.state_[epoch_] = value_.state_[epoch_].array().tanh().matrix();

  const auto& r = receptance_.state_[epoch_];
  const auto& k = key_.state_[epoch_];
  const auto& v = value_.state_[epoch_];

  // WKV update with log-space stabilization
  // decay_log = m_prev + log(exp(-w)) = m_prev - w
  Eigen::ArrayXf decay_log = m_state_.array() - w_.array();
  Eigen::ArrayXf m_new = decay_log.max(k.array());

  Eigen::ArrayXf exp_decay = (decay_log - m_new).exp();
  Eigen::ArrayXf exp_key = (k.array() - m_new).exp();

  saved_exp_decay_[epoch_] = exp_decay.matrix();
  saved_exp_key_[epoch_] = exp_key.matrix();

  // Update WKV state: a = numerator (weighted values), b = denominator (weights)
  a_state_ = (a_state_.array() * exp_decay + exp_key * v.array()).matrix();
  b_state_ = (b_state_.array() * exp_decay + exp_key).matrix();
  m_state_ = m_new.matrix();

  // Normalized output: wkv = a / max(b, eps)
  Eigen::ArrayXf b_safe = b_state_.array().max(1e-6f);
  Eigen::VectorXf wkv = (a_state_.array() / b_safe).matrix();
  saved_wkv_[epoch_] = wkv;

  // Apply receptance gate: output = sigmoid(r) * wkv
  hidden->segment(hidden_start, num_cells_) = r.cwiseProduct(wkv);

  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void RwkvLayer::ForwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf& input, int input_symbol) {
  // Same as LSTM: embedding lookup + matrix multiply + layer norm
  neurons.norm_[epoch_].noalias() = neurons.weights_.col(input_symbol);
  int input_size = input.size();
  neurons.norm_[epoch_].noalias() +=
      neurons.weights_.middleCols(output_size_, input_size) * input;

  float inv_std = 1.0f / sqrtf(
      neurons.norm_[epoch_].squaredNorm() / num_cells_ + 1e-5f);
  neurons.ivar_[epoch_] = inv_std;
  neurons.norm_[epoch_] *= inv_std;
  neurons.state_[epoch_] = (neurons.norm_[epoch_].array() *
      neurons.gamma_.array() + neurons.beta_.array()).matrix();
}

inline void RwkvLayer::ClipGradients(Eigen::VectorXf* arr) {
  *arr = arr->array().max(-gradient_clip_).min(gradient_clip_).matrix();
}

inline void RwkvLayer::BackwardPass(const Eigen::VectorXf& input, int epoch,
    int layer, int input_symbol, Eigen::VectorXf* hidden_error,
    int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    a_error_.setZero();
    b_error_.setZero();
    w_grad_.setZero();
  } else {
    stored_error_ += *hidden_error;
  }

  // Restore saved forward-pass values
  const auto& r = receptance_.state_[epoch];       // sigmoid(r)
  const auto& k = key_.state_[epoch];              // linear k
  const auto& v = value_.state_[epoch];            // tanh(v)
  const auto& wkv = saved_wkv_[epoch];
  const auto& a_prev = saved_a_[epoch];
  const auto& b_prev = saved_b_[epoch];
  const auto& exp_decay = saved_exp_decay_[epoch];
  const auto& exp_key = saved_exp_key_[epoch];

  // Reconstruct current a, b from saved pre-update state
  Eigen::ArrayXf a_cur = a_prev.array() * exp_decay.array() +
      exp_key.array() * v.array();
  Eigen::ArrayXf b_cur = b_prev.array() * exp_decay.array() +
      exp_key.array();

  // output = sigmoid(r) * wkv, where wkv = a / max(b, eps)
  // d_wkv = stored_error * r
  Eigen::ArrayXf d_wkv = stored_error_.array() * r.array();

  // Gradients through wkv = a / b
  Eigen::ArrayXf b_safe = b_cur.max(1e-6f);
  Eigen::ArrayXf d_a = d_wkv / b_safe;
  Eigen::ArrayXf d_b = -d_wkv * a_cur / (b_safe * b_safe);

  // Add state errors propagated from future timesteps
  d_a += a_error_.array();
  d_b += b_error_.array();

  // Receptance error: sigmoid derivative = r * (1 - r)
  receptance_.error_ = (stored_error_.array() * wkv.array() *
      r.array() * (1.0f - r.array())).matrix();

  // Value error: d/dv = d_a * exp_key, tanh derivative = (1 - v^2)
  value_.error_ = (d_a * exp_key.array() *
      (1.0f - v.array().square())).matrix();

  // Key error: d/dk = (d_a * v + d_b) * exp_key
  key_.error_ = ((d_a * v.array() + d_b) * exp_key.array()).matrix();

  // w gradient: d/dw = -(d_a * a_prev + d_b * b_prev) * exp_decay
  w_grad_ += (-(d_a * a_prev.array() + d_b * b_prev.array()) *
      exp_decay.array()).matrix();

  hidden_error->setZero();

  if (epoch > bptt_start) {
    // Propagate WKV state error to previous timestep
    a_error_ = (d_a * exp_decay.array()).matrix();
    b_error_ = (d_b * exp_decay.array()).matrix();
    stored_error_.setZero();
  } else {
    a_error_.setZero();
    b_error_.setZero();
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
      beta1_power_ *= 0.025f;
      beta2_power_ *= 0.9999f;
    }
  }

  // Backward through NeuronLayers (weight updates at bptt_start)
  BackwardPass(receptance_, input, epoch, layer, input_symbol,
      hidden_error, bptt_start);
  BackwardPass(key_, input, epoch, layer, input_symbol,
      hidden_error, bptt_start);
  BackwardPass(value_, input, epoch, layer, input_symbol,
      hidden_error, bptt_start);

  // Adam update for learned decay parameter w at bptt_start
  if (epoch == bptt_start) {
    float bias_corr1 = 1.0f - beta1_power_;
    float bias_corr2 = 1.0f - beta2_power_;
    ClipGradients(&w_grad_);
    Adam(&w_grad_, &w_m_, &w_v_, &w_, learning_rate_, update_steps_,
        bias_corr1, bias_corr2);
    // Clamp decay to stable range: exp(-0.01)≈0.99 to exp(-10)≈0.00005
    w_ = w_.array().max(0.01f).min(10.0f).matrix();
  }

  ClipGradients(&a_error_);
  ClipGradients(&b_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void RwkvLayer::BackwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf& input, int epoch, int layer, int input_symbol,
    Eigen::VectorXf* hidden_error, int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_.setZero();
    neurons.beta_u_.setZero();
    neurons.update_.setZero();
  }

  // Layer norm backward
  neurons.beta_u_ += neurons.error_;
  neurons.gamma_u_ += neurons.error_.cwiseProduct(neurons.norm_[epoch]);
  neurons.error_ = neurons.error_.cwiseProduct(neurons.gamma_) *
      neurons.ivar_[epoch];
  neurons.error_ -= ((neurons.error_.cwiseProduct(neurons.norm_[epoch]).sum()
      / num_cells_) * neurons.norm_[epoch]);

  const int embed_cols = output_size_;
  const int in_cols = input_size_;
  const int offset = embed_cols + in_cols;

  // Inter-layer error propagation
  if (layer > 0) {
    auto W_prev = neurons.weights_.block(0, offset + num_cells_,
        num_cells_, num_cells_);
    hidden_error->segment(0, num_cells_).noalias() +=
        W_prev.transpose() * neurons.error_;
  }

  // Recurrent error accumulation
  if (epoch > bptt_start) {
    auto W_rec = neurons.weights_.block(0, offset, num_cells_, num_cells_);
    stored_error_.noalias() += W_rec.transpose() * neurons.error_;
  }

  // Weight gradient accumulation
  int input_size = input.size();
  neurons.update_.middleCols(output_size_, input_size).noalias() +=
      neurons.error_ * input.transpose();
  neurons.update_.col(input_symbol) += neurons.error_;

  if (epoch == bptt_start) {
    float bias_corr1 = 1.0f - beta1_power_;
    float bias_corr2 = 1.0f - beta2_power_;
    AdamMatrix(&neurons.update_, &neurons.m_, &neurons.v_,
        &neurons.weights_, learning_rate_, update_steps_,
        bias_corr1, bias_corr2);
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_,
        bias_corr1, bias_corr2);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_,
        bias_corr1, bias_corr2);
  }
}

inline std::vector<Eigen::MatrixXf*> RwkvLayer::Weights() {
  std::vector<Eigen::MatrixXf*> weights;
  weights.push_back(&receptance_.weights_);
  weights.push_back(&key_.weights_);
  weights.push_back(&value_.weights_);
  return weights;
}
