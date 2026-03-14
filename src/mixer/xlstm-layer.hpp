#include "xlstm-layer.h"

#include <math.h>
#include <algorithm>
#include "../random.hpp"

inline XLstmLayer::XLstmLayer(unsigned int input_size,
    unsigned int auxiliary_input_size, unsigned int output_size,
    unsigned int num_cells, int horizon, float gradient_clip,
    float learning_rate) :
    cell_state_(Eigen::VectorXf::Zero(num_cells)),
    normalizer_(Eigen::VectorXf::Zero(num_cells)),
    m_state_(Eigen::VectorXf::Zero(num_cells)),
    cell_error_(Eigen::VectorXf::Zero(num_cells)),
    norm_error_(Eigen::VectorXf::Zero(num_cells)),
    stored_error_(Eigen::VectorXf::Zero(num_cells)),
    prev_cell_(horizon, Eigen::VectorXf::Zero(num_cells)),
    prev_norm_(horizon, Eigen::VectorXf::Zero(num_cells)),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    input_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    output_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    candidate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;

  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_.cols(); ++j) {
      forget_gate_.weights_(i, j) = low + get_uniform_random() * range;
      input_gate_.weights_(i, j) = low + get_uniform_random() * range;
      output_gate_.weights_(i, j) = low + get_uniform_random() * range;
      candidate_.weights_(i, j) = low + get_uniform_random() * range;
    }
    // Bias forget gate toward remembering (same convention as LSTM)
    forget_gate_.weights_(i, forget_gate_.weights_.cols() - 1) = 1;
  }
}

inline void XLstmLayer::ForwardPass(const Eigen::VectorXf& input,
    int input_symbol, Eigen::VectorXf* hidden, int hidden_start) {
  // Save previous state for BPTT
  prev_cell_[epoch_] = cell_state_;
  prev_norm_[epoch_] = normalizer_;

  // Compute all 4 gate pre-activations (layer-normed)
  ForwardPassGate(forget_gate_, input, input_symbol);
  ForwardPassGate(input_gate_, input, input_symbol);
  ForwardPassGate(output_gate_, input, input_symbol);
  ForwardPassGate(candidate_, input, input_symbol);

  // --- Exponential gating with log-space stabilization ---
  // log_f = pre-activation of forget gate, log_i = pre-activation of input gate
  const auto& log_f = forget_gate_.state_[epoch_];
  const auto& log_i = input_gate_.state_[epoch_];

  // m_t = max(log_f + m_{t-1}, log_i) element-wise
  Eigen::VectorXf m_prev = m_state_;
  m_state_ = (log_f.array() + m_prev.array()).max(log_i.array()).matrix();

  // Stabilized gates: f' = exp(log_f + m_prev - m), i' = exp(log_i - m)
  forget_gate_.state_[epoch_] =
      ((log_f.array() + m_prev.array() - m_state_.array()).exp()).matrix();
  input_gate_.state_[epoch_] =
      ((log_i.array() - m_state_.array()).exp()).matrix();

  // Standard activations for output (sigmoid) and candidate (tanh)
  output_gate_.state_[epoch_] =
      (1.0f / (1.0f + (-output_gate_.state_[epoch_].array()).exp())).matrix();
  candidate_.state_[epoch_] =
      candidate_.state_[epoch_].array().tanh().matrix();

  const auto& f = forget_gate_.state_[epoch_];
  const auto& ip = input_gate_.state_[epoch_];
  const auto& z = candidate_.state_[epoch_];
  const auto& o = output_gate_.state_[epoch_];

  // Cell state: c = f' * c_prev + i' * z
  cell_state_ = f.cwiseProduct(prev_cell_[epoch_]) + ip.cwiseProduct(z);

  // Normalizer: n = f' * n_prev + i'
  normalizer_ = f.cwiseProduct(prev_norm_[epoch_]) + ip;

  // Output: h = o * (c / max(|n|, 1))  — normalizer replaces tanh
  Eigen::ArrayXf D = normalizer_.array().abs().max(1.0f);
  hidden->segment(hidden_start, num_cells_) =
      o.cwiseProduct((cell_state_.array() / D).matrix());

  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void XLstmLayer::ForwardPassGate(NeuronLayer& neurons,
    const Eigen::VectorXf& input, int input_symbol) {
  neurons.norm_[epoch_].noalias() = neurons.weights_.col(input_symbol);
  int input_size = input.size();
  neurons.norm_[epoch_].noalias() +=
      neurons.weights_.middleCols(output_size_, input_size) * input;

  float inv_std = 1.0f / sqrtf(
      neurons.norm_[epoch_].squaredNorm() / num_cells_ + 1e-5f);
  neurons.ivar_[epoch_] = inv_std;
  neurons.norm_[epoch_] *= inv_std;
  neurons.state_[epoch_] =
      (neurons.norm_[epoch_].array() * neurons.gamma_.array() +
       neurons.beta_.array()).matrix();
}

inline void XLstmLayer::ClipGradients(Eigen::VectorXf* arr) {
  *arr = arr->array().max(-gradient_clip_).min(gradient_clip_).matrix();
}

inline void XLstmLayer::BackwardPass(const Eigen::VectorXf& input, int epoch,
    int layer, int input_symbol, Eigen::VectorXf* hidden_error,
    int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    cell_error_.setZero();
    norm_error_.setZero();
  } else {
    stored_error_ += *hidden_error;
  }

  // Retrieve stored gate activations and states
  const auto& f = forget_gate_.state_[epoch];
  const auto& ip = input_gate_.state_[epoch];
  const auto& o = output_gate_.state_[epoch];
  const auto& z = candidate_.state_[epoch];
  const auto& c_prev = prev_cell_[epoch];
  const auto& n_prev = prev_norm_[epoch];

  // Recompute c_t, n_t from stored values
  Eigen::VectorXf c = f.cwiseProduct(c_prev) + ip.cwiseProduct(z);
  Eigen::VectorXf n = f.cwiseProduct(n_prev) + ip;
  Eigen::ArrayXf D = n.array().abs().max(1.0f);
  Eigen::ArrayXf c_hat = c.array() / D;

  // --- Output gate (sigmoid): d_o = d_h * c_hat ---
  output_gate_.error_ = (stored_error_.array() * c_hat *
                         o.array() * (1.0f - o.array())).matrix();

  // --- Gradient through c/D normalization ---
  Eigen::ArrayXf d_c_hat = stored_error_.array() * o.array();
  cell_error_.array() += d_c_hat / D;

  // d_n only when |n| >= 1: d(c/|n|)/dn = -c*n / |n|^3 = -c*n / D^3
  Eigen::ArrayXf D3 = D * D * D;
  Eigen::ArrayXf n_mask = (n.array().abs() >= 1.0f).cast<float>();
  norm_error_.array() -= d_c_hat * c.array() * n.array() * n_mask / D3;

  // --- Candidate (tanh): d_z = d_c * i' * tanh'(z) ---
  candidate_.error_ = (cell_error_.array() * ip.array() *
                       (1.0f - z.array().square())).matrix();

  // --- Input gate (exp, straight-through): d_i_pre = d_i' * i' ---
  // d_i' = d_c * z + d_n (from c = f'*c_prev + i'*z and n = f'*n_prev + i')
  input_gate_.error_ =
      ((cell_error_.array() * z.array() + norm_error_.array()) *
       ip.array()).matrix();

  // --- Forget gate (exp, straight-through): d_f_pre = d_f' * f' ---
  // d_f' = d_c * c_prev + d_n * n_prev
  forget_gate_.error_ =
      ((cell_error_.array() * c_prev.array() +
        norm_error_.array() * n_prev.array()) * f.array()).matrix();

  hidden_error->setZero();

  if (epoch > bptt_start) {
    // Temporal propagation through cell and normalizer state
    cell_error_ = cell_error_.cwiseProduct(f);
    norm_error_ = norm_error_.cwiseProduct(f);
    stored_error_.setZero();
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
      beta1_power_ *= 0.025f;
      beta2_power_ *= 0.9999f;
    }
  }

  // Layer norm backward + weight gradients + recurrent error for all 4 gates
  BackwardPassGate(forget_gate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);
  BackwardPassGate(input_gate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);
  BackwardPassGate(output_gate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);
  BackwardPassGate(candidate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);

  ClipGradients(&cell_error_);
  ClipGradients(&norm_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void XLstmLayer::BackwardPassGate(NeuronLayer& neurons,
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
  neurons.error_ =
      neurons.error_.cwiseProduct(neurons.gamma_) * neurons.ivar_[epoch];
  neurons.error_ -=
      ((neurons.error_.cwiseProduct(neurons.norm_[epoch]).sum() /
        num_cells_) * neurons.norm_[epoch]);

  const int offset = output_size_ + input_size_;

  // Inter-layer error propagation
  if (layer > 0) {
    auto W_prev = neurons.weights_.block(
        0, offset + num_cells_, num_cells_, num_cells_);
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

  // Adam update at BPTT boundary
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

inline std::vector<Eigen::MatrixXf*> XLstmLayer::Weights() {
  return {&forget_gate_.weights_, &input_gate_.weights_,
          &output_gate_.weights_, &candidate_.weights_};
}
