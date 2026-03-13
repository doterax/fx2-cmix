#include "lstm-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>
#include "..\random.hpp"

#define FAST_TANH tanh //fast_tanh
#define FAST_TANH_VEC tanh //fast_tanh_vec
namespace {
// inline float fast_tanh(const float x)
// {
//     const float ax = fabs(x);
//     const float x2 = x * x;

//     return(x * (2.45550750702956f + 2.45550750702956f * ax +
//         (0.893229853513558f + 0.821226666969744f * ax) * x2) /
//         (2.44506634652299f + (2.44506634652299f + x2) *
//             fabs(x + 0.814642734961073f * x * ax)));
// }

// float fast_tanh(float x){
//   float x2 = x * x;
//   float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
//   float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
//   return a / b;
// }
// 
// template <class _Tp>
// inline std::valarray<_Tp> fast_tanh_vec(const std::valarray<_Tp>& __x) {
//   std::valarray<_Tp> __tmp(__x.size());
//   for (size_t __i = 0; __i < __x.size(); ++__i)
//     __tmp[__i] = fast_tanh(__x[__i]);
//   return __tmp;
// }

inline void Adam(Eigen::VectorXf* g, Eigen::VectorXf* m,
    Eigen::VectorXf* v, Eigen::VectorXf* w, float learning_rate,
    float t, float bias_corr1, float bias_corr2) {
  const float eps = 1e-6f; 
  const float beta1 = 0.025f, beta2 = 0.9999f;
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f); 
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f); 
  }
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * (*g).cwiseProduct(*g);
  w->array() -= alpha * (m->array() / bias_corr1) /
      ((v->array() / bias_corr2).sqrt() + eps);
}

inline void AdamMatrix(Eigen::MatrixXf* g, Eigen::MatrixXf* m,
    Eigen::MatrixXf* v, Eigen::MatrixXf* w, float learning_rate,
    float t, float bias_corr1, float bias_corr2) {
  const float eps = 1e-6f;
  const float beta1 = 0.025f, beta2 = 0.9999f;
  
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f); 
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f); 
  }
  
  // Match original Adam exactly
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * g->cwiseProduct(*g);
  
  w->array() -= alpha * (m->array() / bias_corr1) /
      ((v->array() / bias_corr2).sqrt() + eps);
}

}

inline LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    state_(Eigen::VectorXf::Zero(num_cells)), 
    state_error_(Eigen::VectorXf::Zero(num_cells)), 
    stored_error_(Eigen::VectorXf::Zero(num_cells)),
    tanh_state_(horizon, Eigen::VectorXf::Zero(num_cells)),
    input_gate_state_(horizon, Eigen::VectorXf::Zero(num_cells)),
    last_state_(horizon, Eigen::VectorXf::Zero(num_cells)),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
    output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  
  // Initialize weight matrices with Xavier initialization
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_.cols(); ++j) {
      forget_gate_.weights_(i, j) = low + get_uniform_random() * range;
      input_node_.weights_(i, j) = low + get_uniform_random() * range;
      output_gate_.weights_(i, j) = low + get_uniform_random() * range;
    }
    forget_gate_.weights_(i, forget_gate_.weights_.cols() - 1) = 1;
  }
}

inline void LstmLayer::ForwardPass(const Eigen::VectorXf& input, int input_symbol,
    Eigen::VectorXf* hidden, int hidden_start) {
  last_state_[epoch_] = state_;
  ForwardPass(forget_gate_, input, input_symbol);
  ForwardPass(input_node_, input, input_symbol);
  ForwardPass(output_gate_, input, input_symbol);
  
  // Vectorized activation functions using Eigen's array operations
  forget_gate_.state_[epoch_] = (1.0f / (1.0f + (-forget_gate_.state_[epoch_].array()).exp())).matrix();
  input_node_.state_[epoch_] = input_node_.state_[epoch_].array().tanh().matrix();
  output_gate_.state_[epoch_] = (1.0f / (1.0f + (-output_gate_.state_[epoch_].array()).exp())).matrix();
  
  // Vectorized gate computations
  input_gate_state_[epoch_] = (1.0f - forget_gate_.state_[epoch_].array()).matrix();
  state_ = state_.cwiseProduct(forget_gate_.state_[epoch_]);
  state_ += input_node_.state_[epoch_].cwiseProduct(input_gate_state_[epoch_]);
  tanh_state_[epoch_] = state_.array().tanh().matrix();
  hidden->segment(hidden_start, num_cells_) = output_gate_.state_[epoch_].cwiseProduct(tanh_state_[epoch_]);
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf& input, int input_symbol) {
  // Batched matrix-vector multiply: W * [input_embedding; input_features]
  // Extract input symbol column + matrix multiply for continuous input
  neurons.norm_[epoch_].noalias() = neurons.weights_.col(input_symbol);
  int input_size = input.size();
  neurons.norm_[epoch_].noalias() += neurons.weights_.middleCols(output_size_, input_size) * input;
  
  // Layer normalization: fused normalize + affine transform
  float inv_std = 1.0f / sqrtf(neurons.norm_[epoch_].squaredNorm() / num_cells_ + 1e-5f);
  neurons.ivar_[epoch_] = inv_std;
  neurons.norm_[epoch_] *= inv_std;
  neurons.state_[epoch_] = (neurons.norm_[epoch_].array() * neurons.gamma_.array() + neurons.beta_.array()).matrix();
}

inline void LstmLayer::ClipGradients(Eigen::VectorXf* arr) {
  // Vectorized gradient clipping using Eigen's array operations
  *arr = arr->array().max(-gradient_clip_).min(gradient_clip_).matrix();
}

inline void LstmLayer::BackwardPass(const Eigen::VectorXf&input, int epoch,
    int layer, int input_symbol, Eigen::VectorXf* hidden_error,
    int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_.setZero();
  } else {
    stored_error_ += *hidden_error;
  }

  // Vectorized LSTM gate error gradients
  // Output gate: derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
  output_gate_.error_ = (tanh_state_[epoch].array() * stored_error_.array() * 
                         output_gate_.state_[epoch].array() * 
                         (1.0f - output_gate_.state_[epoch].array())).matrix();
  
  // State error: derivative of tanh(x) is (1 - tanh²(x))
  state_error_ += (stored_error_.array() * output_gate_.state_[epoch].array() * 
                   (1.0f - tanh_state_[epoch].array().square())).matrix();
  
  // Input node: derivative of tanh(x) is (1 - tanh²(x))
  input_node_.error_ = (state_error_.array() * input_gate_state_[epoch].array() * 
                        (1.0f - input_node_.state_[epoch].array().square())).matrix();
  
  // Forget gate: derivative through state computation
  forget_gate_.error_ = ((last_state_[epoch] - input_node_.state_[epoch]).array() *
                         state_error_.array() * forget_gate_.state_[epoch].array() * 
                         input_gate_state_[epoch].array()).matrix();

  hidden_error->setZero();
  if (epoch > bptt_start) {
    state_error_ = state_error_.cwiseProduct(forget_gate_.state_[epoch]);
    stored_error_.setZero();
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
      beta1_power_ *= 0.025f;
      beta2_power_ *= 0.9999f;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error, bptt_start);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error, bptt_start);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error, bptt_start);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void LstmLayer::BackwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf&input, int epoch, int layer, int input_symbol,
    Eigen::VectorXf* hidden_error, int bptt_start) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_.setZero();
    neurons.beta_u_.setZero();
    neurons.update_.setZero();
  }
  
  neurons.beta_u_ += neurons.error_;
  neurons.gamma_u_ += neurons.error_.cwiseProduct(neurons.norm_[epoch]);
  neurons.error_ = neurons.error_.cwiseProduct(neurons.gamma_) * neurons.ivar_[epoch];
  neurons.error_ -= ((neurons.error_.cwiseProduct(neurons.norm_[epoch]).sum() / num_cells_) * neurons.norm_[epoch]);
  
  // Strict layout partitions to match original indexing
  const int embed_cols = output_size_;
  const int in_cols = input_size_;
  const int offset = embed_cols + in_cols; // start of temporal+inter-layer blocks
  // recurrent block: columns [offset .. offset+num_cells_-1]
  // prev-layer block: columns [offset+num_cells_ .. offset+2*num_cells_-1]

  // Hidden layer error propagation (to previous layer)
  if (layer > 0) {
    auto W_prev = neurons.weights_.block(0, offset + num_cells_, num_cells_, num_cells_);
    hidden_error->segment(0, num_cells_).noalias() += W_prev.transpose() * neurons.error_;
  }

  // Recurrent error accumulation (to previous time step)
  if (epoch > bptt_start) {
    auto W_rec = neurons.weights_.block(0, offset, num_cells_, num_cells_);
    stored_error_.noalias() += W_rec.transpose() * neurons.error_;
  }
  
  // Batched outer product for weight gradient: error * input^T
  int input_size = input.size();
  neurons.update_.middleCols(output_size_, input_size).noalias() += neurons.error_ * input.transpose();
  neurons.update_.col(input_symbol) += neurons.error_;
  
  if (epoch == bptt_start) {
    // Use pre-computed running power products for bias correction
    float bias_corr1 = 1.0f - beta1_power_;
    float bias_corr2 = 1.0f - beta2_power_;
    // Matrix-wise Adam update
    AdamMatrix(&neurons.update_, &neurons.m_, &neurons.v_, &neurons.weights_, learning_rate_, update_steps_, bias_corr1, bias_corr2);
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_, &neurons.gamma_, learning_rate_, update_steps_, bias_corr1, bias_corr2);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_, &neurons.beta_, learning_rate_, update_steps_, bias_corr1, bias_corr2);
  }
}

inline std::vector<Eigen::MatrixXf*> LstmLayer::Weights() {
  std::vector<Eigen::MatrixXf*> weights;
  weights.push_back(&forget_gate_.weights_);
  weights.push_back(&input_node_.weights_);
  weights.push_back(&output_gate_.weights_);
  return weights;
}

