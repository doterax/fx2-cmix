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
    float t) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f; 
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
  if (t < UPDATE_LIMIT) {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))).cwiseQuotient(
        ((*v) / (float)(1.0f - pow(beta2, t))).array().sqrt().matrix() + eps * Eigen::VectorXf::Ones(w->size())));
  } else {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, UPDATE_LIMIT))).cwiseQuotient(
        ((*v) / (float)(1.0f - pow(beta2, UPDATE_LIMIT))).array().sqrt().matrix() + eps * Eigen::VectorXf::Ones(w->size())));
  }
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
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_[i].size(); ++j) {
      forget_gate_.weights_[i][j] = low + get_uniform_random() * range;
      input_node_.weights_[i][j] = low + get_uniform_random() * range;
      output_gate_.weights_[i][j] = low + get_uniform_random() * range;
    }
    forget_gate_.weights_[i][forget_gate_.weights_[i].size() - 1] = 1;
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
  input_gate_state_[epoch_] = Eigen::VectorXf::Ones(num_cells_) - forget_gate_.state_[epoch_];
  state_ = state_.cwiseProduct(forget_gate_.state_[epoch_]);
  state_ += input_node_.state_[epoch_].cwiseProduct(input_gate_state_[epoch_]);
  tanh_state_[epoch_] = state_.array().tanh().matrix();
  hidden->segment(hidden_start, num_cells_) = output_gate_.state_[epoch_].cwiseProduct(tanh_state_[epoch_]);
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf& input, int input_symbol) {
  // Vectorized computation: accumulate input_symbol contribution
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.norm_[epoch_][i] = neurons.weights_[i][input_symbol];
  }
  
  // Vectorized matrix-vector multiplication for input contribution
  int input_size = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.norm_[epoch_][i] += neurons.weights_[i].segment(output_size_, input_size).dot(input);
  }
  
  // Layer normalization with vectorized operations
  float variance = neurons.norm_[epoch_].squaredNorm() / num_cells_;
  neurons.ivar_[epoch_] = 1.0f / sqrt(variance + 1e-5f);
  neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
  neurons.state_[epoch_] = neurons.norm_[epoch_].cwiseProduct(neurons.gamma_) +
      neurons.beta_;
}

inline void LstmLayer::ClipGradients(Eigen::VectorXf* arr) {
  // Vectorized gradient clipping using Eigen's array operations
  *arr = arr->array().max(-gradient_clip_).min(gradient_clip_).matrix();
}

inline void LstmLayer::BackwardPass(const Eigen::VectorXf&input, int epoch,
    int layer, int input_symbol, Eigen::VectorXf* hidden_error) {
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
  if (epoch > 0) {
    state_error_ = state_error_.cwiseProduct(forget_gate_.state_[epoch]);
    stored_error_.setZero();
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void LstmLayer::BackwardPass(NeuronLayer& neurons,
    const Eigen::VectorXf&input, int epoch, int layer, int input_symbol,
    Eigen::VectorXf* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_.setZero();
    neurons.beta_u_.setZero();
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i].setZero();
      int offset = output_size_ + input_size_;
      for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
        neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
      }
    }
  }
  neurons.beta_u_ += neurons.error_;
  neurons.gamma_u_ += neurons.error_.cwiseProduct(neurons.norm_[epoch]);
  neurons.error_ = neurons.error_.cwiseProduct(neurons.gamma_) * neurons.ivar_[epoch];
  neurons.error_ -= ((neurons.error_.cwiseProduct(neurons.norm_[epoch]).sum() /
      num_cells_) * neurons.norm_[epoch]);
  
  // Vectorized backward pass through hidden layer connections
  if (layer > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      // Use dot product for vectorized computation
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
      }
      (*hidden_error)[i] += f;
    }
  }
  
  // Vectorized recurrent error accumulation
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      // Use dot product for vectorized computation
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[i][j];
      }
      stored_error_[i] += f;
    }
  }
  
  // Vectorized weight gradient accumulation
  int input_size = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    // Outer product: error[i] * input -> gradient contribution
    neurons.update_[i].segment(output_size_, input_size) += neurons.error_[i] * input;
    neurons.update_[i][input_symbol] += neurons.error_[i];
  }
  if (epoch == 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
          &neurons.weights_[i], learning_rate_, update_steps_);
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_);
  }
}

inline std::vector<std::vector<Eigen::VectorXf>*> LstmLayer::Weights() {
  std::vector<std::vector<Eigen::VectorXf>*> weights;
  weights.push_back(&forget_gate_.weights_);
  weights.push_back(&input_node_.weights_);
  weights.push_back(&output_gate_.weights_);
  return weights;
}

