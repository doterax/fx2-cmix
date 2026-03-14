#include "gru-layer.h"

#include <math.h>
#include <algorithm>
#include <numeric>
#include "../random.hpp"

// Adam / AdamMatrix are already available via lstm-layer.h -> lstm-layer.hpp
// (defined in an unnamed namespace there).

inline GruLayer::GruLayer(unsigned int input_size,
    unsigned int auxiliary_input_size, unsigned int output_size,
    unsigned int num_cells, int horizon, float gradient_clip,
    float learning_rate) :
    stored_error_(Eigen::VectorXf::Zero(num_cells)),
    last_hidden_(horizon, Eigen::VectorXf::Zero(num_cells)),
    reset_hidden_(horizon, Eigen::VectorXf::Zero(num_cells)),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    reset_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    update_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    candidate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;

  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < reset_gate_.weights_.cols(); ++j) {
      reset_gate_.weights_(i, j) = low + get_uniform_random() * range;
      update_gate_.weights_(i, j) = low + get_uniform_random() * range;
      candidate_.weights_(i, j) = low + get_uniform_random() * range;
    }
    // Bias update gate toward 1 (= keep old hidden state initially),
    // analogous to LSTM forget gate bias.
    update_gate_.weights_(i, update_gate_.weights_.cols() - 1) = 1;
  }
}

inline void GruLayer::ForwardPass(const Eigen::VectorXf& input,
    int input_symbol, Eigen::VectorXf* hidden, int hidden_start) {
  last_hidden_[epoch_] = hidden->segment(hidden_start, num_cells_);

  // Reset gate: r = sigmoid(LayerNorm(W_r * input))
  ForwardPassGate(reset_gate_, input, input_symbol);
  reset_gate_.state_[epoch_] =
      (1.0f / (1.0f + (-reset_gate_.state_[epoch_].array()).exp())).matrix();

  // Update gate: z = sigmoid(LayerNorm(W_z * input))
  ForwardPassGate(update_gate_, input, input_symbol);
  update_gate_.state_[epoch_] =
      (1.0f / (1.0f + (-update_gate_.state_[epoch_].array()).exp())).matrix();

  // Candidate: n = tanh(LayerNorm(W_n * [x, r*h_prev, ...]))
  Eigen::VectorXf mod_input = input;
  mod_input.segment(input_size_, num_cells_) =
      reset_gate_.state_[epoch_].cwiseProduct(
          input.segment(input_size_, num_cells_));
  reset_hidden_[epoch_] = mod_input.segment(input_size_, num_cells_);
  ForwardPassGate(candidate_, mod_input, input_symbol);
  candidate_.state_[epoch_] =
      candidate_.state_[epoch_].array().tanh().matrix();

  // h = z * h_prev + (1-z) * n
  hidden->segment(hidden_start, num_cells_) =
      update_gate_.state_[epoch_].cwiseProduct(last_hidden_[epoch_]) +
      (1.0f - update_gate_.state_[epoch_].array()).matrix()
          .cwiseProduct(candidate_.state_[epoch_]);

  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void GruLayer::ForwardPassGate(NeuronLayer& neurons,
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

inline void GruLayer::ClipGradients(Eigen::VectorXf* arr) {
  *arr = arr->array().max(-gradient_clip_).min(gradient_clip_).matrix();
}

inline void GruLayer::BackwardPass(const Eigen::VectorXf& input, int epoch,
    int layer, int input_symbol, Eigen::VectorXf* hidden_error,
    int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
  } else {
    stored_error_ += *hidden_error;
  }

  const auto& z = update_gate_.state_[epoch];
  const auto& n = candidate_.state_[epoch];
  const auto& r = reset_gate_.state_[epoch];
  const auto& h_prev = last_hidden_[epoch];

  // --- Gate errors (including activation derivatives) ---

  // Update gate: dh*(h_prev - n) * sigmoid'(z)
  update_gate_.error_ = (stored_error_.array() * (h_prev - n).array() *
                         z.array() * (1.0f - z.array())).matrix();

  // Candidate: dh*(1-z) * tanh'(n)
  candidate_.error_ = (stored_error_.array() * (1.0f - z.array()) *
                       (1.0f - n.array().square())).matrix();

  // --- Candidate layer norm backward (inline) ---
  if (epoch == (int)horizon_ - 1) {
    candidate_.gamma_u_.setZero();
    candidate_.beta_u_.setZero();
    candidate_.update_.setZero();
  }
  candidate_.beta_u_ += candidate_.error_;
  candidate_.gamma_u_ +=
      candidate_.error_.cwiseProduct(candidate_.norm_[epoch]);
  candidate_.error_ =
      candidate_.error_.cwiseProduct(candidate_.gamma_) *
      candidate_.ivar_[epoch];
  candidate_.error_ -=
      ((candidate_.error_.cwiseProduct(candidate_.norm_[epoch]).sum() /
        num_cells_) * candidate_.norm_[epoch]);
  // candidate_.error_ now = gradient w.r.t. pre-norm (W_n * mod_input)

  // --- Reset gate error (from candidate gradient through mod_input) ---
  int offset = output_size_ + input_size_;
  auto W_n_rec = candidate_.weights_.block(0, offset,
                                           num_cells_, num_cells_);
  Eigen::VectorXf d_mod_hidden =
      W_n_rec.transpose() * candidate_.error_;

  // dr = d_mod_hidden * h_prev * sigmoid'(r)
  reset_gate_.error_ = (d_mod_hidden.array() * h_prev.array() *
                        r.array() * (1.0f - r.array())).matrix();

  // --- Candidate weight gradient (uses mod_input, not original) ---
  Eigen::VectorXf mod_input = input;
  mod_input.segment(input_size_, num_cells_) = reset_hidden_[epoch];
  int input_size = mod_input.size();
  candidate_.update_.middleCols(output_size_, input_size).noalias() +=
      candidate_.error_ * mod_input.transpose();
  candidate_.update_.col(input_symbol) += candidate_.error_;

  // --- Set up recurrent + inter-layer propagation ---
  hidden_error->setZero();

  if (epoch > bptt_start) {
    // Structural gradients to h_prev (not through weight matrices):
    //   direct: dh * z
    //   candidate: d_mod_hidden * r
    stored_error_ = stored_error_.cwiseProduct(z);
    stored_error_ += d_mod_hidden.cwiseProduct(r);
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
      beta1_power_ *= 0.025f;
      beta2_power_ *= 0.9999f;
    }
  }

  // --- Update and reset gate backward (standard path) ---
  BackwardPassGate(update_gate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);
  BackwardPassGate(reset_gate_, input, epoch, layer, input_symbol,
                   hidden_error, bptt_start);

  // --- Candidate inter-layer gradient ---
  if (layer > 0) {
    auto W_n_prev = candidate_.weights_.block(
        0, offset + num_cells_, num_cells_, num_cells_);
    hidden_error->segment(0, num_cells_).noalias() +=
        W_n_prev.transpose() * candidate_.error_;
  }

  // --- Candidate Adam update ---
  if (epoch == bptt_start) {
    float bias_corr1 = 1.0f - beta1_power_;
    float bias_corr2 = 1.0f - beta2_power_;
    AdamMatrix(&candidate_.update_, &candidate_.m_, &candidate_.v_,
               &candidate_.weights_, learning_rate_, update_steps_,
               bias_corr1, bias_corr2);
    Adam(&candidate_.gamma_u_, &candidate_.gamma_m_, &candidate_.gamma_v_,
         &candidate_.gamma_, learning_rate_, update_steps_,
         bias_corr1, bias_corr2);
    Adam(&candidate_.beta_u_, &candidate_.beta_m_, &candidate_.beta_v_,
         &candidate_.beta_, learning_rate_, update_steps_,
         bias_corr1, bias_corr2);
  }

  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void GruLayer::BackwardPassGate(NeuronLayer& neurons,
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

  int offset = output_size_ + input_size_;

  // Inter-layer error propagation
  if (layer > 0) {
    auto W_prev = neurons.weights_.block(
        0, offset + num_cells_, num_cells_, num_cells_);
    hidden_error->segment(0, num_cells_).noalias() +=
        W_prev.transpose() * neurons.error_;
  }

  // Recurrent error propagation
  if (epoch > bptt_start) {
    auto W_rec = neurons.weights_.block(0, offset, num_cells_, num_cells_);
    stored_error_.noalias() += W_rec.transpose() * neurons.error_;
  }

  // Weight gradient accumulation
  int input_size = input.size();
  neurons.update_.middleCols(output_size_, input_size).noalias() +=
      neurons.error_ * input.transpose();
  neurons.update_.col(input_symbol) += neurons.error_;

  // Adam update at bptt boundary
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

inline std::vector<Eigen::MatrixXf*> GruLayer::Weights() {
  return {&reset_gate_.weights_, &update_gate_.weights_,
          &candidate_.weights_};
}
