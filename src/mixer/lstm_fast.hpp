#ifndef LSTM_FAST_HPP
#define LSTM_FAST_HPP

#include "lstm_fast.h"

#include <algorithm>
#include <cmath>
#include "../random.hpp"

namespace lstm_fast_detail {

inline EIGEN_STRONG_INLINE float AdamAlpha(float lr, unsigned long long t) {
  const float limit = static_cast<float>(UPDATE_LIMIT);
  float tt = (t < UPDATE_LIMIT) ? static_cast<float>(t) : limit;
  return lr * 0.1f / std::sqrt(5e-5f * tt + 1.0f);
}

// Vector Adam update, matching Lstm's Adam exactly.
inline EIGEN_STRONG_INLINE void AdamVec(Eigen::VectorXf& g, Eigen::VectorXf& m,
                                        Eigen::VectorXf& v, Eigen::VectorXf& w,
                                        float alpha, float bc1, float bc2) {
  const float eps   = 1e-6f;
  const float beta1 = 0.025f, beta2 = 0.9999f;
  m.array() = beta1 * m.array() + (1.0f - beta1) * g.array();
  v.array() = beta2 * v.array() + (1.0f - beta2) * g.array().square();
  w.array() -= alpha * (m.array() / bc1) /
               ((v.array() / bc2).sqrt() + eps);
}

// Matrix Adam update, matching AdamMatrix exactly.
inline EIGEN_STRONG_INLINE void AdamMat(Eigen::MatrixXf& g, Eigen::MatrixXf& m,
                                        Eigen::MatrixXf& v, Eigen::MatrixXf& w,
                                        float alpha, float bc1, float bc2) {
  const float eps   = 1e-6f;
  const float beta1 = 0.025f, beta2 = 0.9999f;
  m.array() = beta1 * m.array() + (1.0f - beta1) * g.array();
  v.array() = beta2 * v.array() + (1.0f - beta2) * g.array().square();
  w.array() -= alpha * (m.array() / bc1) /
               ((v.array() / bc2).sqrt() + eps);
}

}  // namespace lstm_fast_detail

inline LstmFast::LstmFast(unsigned int input_size, unsigned int output_size,
                        unsigned int num_cells, unsigned int num_layers,
                        int horizon, float learning_rate, float gradient_clip,
                        int bptt_depth, int bptt_period)
    : input_history_(horizon),
      learning_rate_(learning_rate),
      gradient_clip_(gradient_clip),
      num_cells_(num_cells),
      input_size_(input_size),
      output_size_(output_size),
      num_layers_(num_layers),
      horizon_(horizon),
      bptt_depth_(bptt_depth <= 0 ? horizon : std::min(bptt_depth, horizon)),
      bptt_period_(bptt_period < 1 ? 1 : bptt_period) {
  layer_input_dim_ = input_size_ + 2u * num_cells_ + 1u;
  total_cols_      = output_size_ + layer_input_dim_;

  const int C  = static_cast<int>(num_cells_);
  const int TC = 3 * C;
  const int H  = horizon_;

  hidden_       = Eigen::VectorXf::Zero(num_cells_ * num_layers_ + 1);
  hidden_[hidden_.size() - 1] = 1.0f;
  hidden_error_ = Eigen::VectorXf::Zero(num_cells_);
  output_layer_ = Eigen::MatrixXf::Zero(num_cells_ * num_layers_ + 1, output_size_);
  output_current_ = Eigen::VectorXf::Constant(output_size_, 1.0f / output_size_);
  hidden_ring_  = Eigen::MatrixXf::Zero(num_cells_ * num_layers_ + 1, H);
  // errors_ring_ column k encodes (output_history_.col(k-1 mod H) -
  // onehot(input_history_[k-1 mod H])) * learning_rate_. In the original
  // startup state output_history_ was uniform 1/V and input_history_ was 0,
  // so the equivalent initial value is (1/V * ones - onehot(0)) * lr for
  // every column. This keeps warm-up behaviour bit-compatible with the
  // old output_history_-based code.
  errors_ring_  = Eigen::MatrixXf::Constant(output_size_, H,
                                            learning_rate_ / output_size_);
  errors_ring_.row(0).array() -= learning_rate_;

  layer_inputs_.resize(num_layers_);
  for (unsigned int l = 0; l < num_layers_; ++l) {
    layer_inputs_[l] = Eigen::MatrixXf::Zero(layer_input_dim_, H);
    // Bias row = 1 for every epoch.
    layer_inputs_[l].row(layer_input_dim_ - 1).setOnes();
  }

  layers_.resize(num_layers_);
  const float val   = std::sqrt(6.0f / float(input_size_ + output_size_));
  const float low   = -val;
  const float range = 2.0f * val;

  for (unsigned int l = 0; l < num_layers_; ++l) {
    Layer& L = layers_[l];
    L.W        = Eigen::MatrixXf::Zero(TC, total_cols_);
    L.W_update = Eigen::MatrixXf::Zero(TC, total_cols_);
    L.W_m      = Eigen::MatrixXf::Zero(TC, total_cols_);
    L.W_v      = Eigen::MatrixXf::Zero(TC, total_cols_);

    L.gamma   = Eigen::VectorXf::Ones(TC);
    L.beta    = Eigen::VectorXf::Zero(TC);
    L.gamma_u = Eigen::VectorXf::Zero(TC);
    L.gamma_m = Eigen::VectorXf::Zero(TC);
    L.gamma_v = Eigen::VectorXf::Zero(TC);
    L.beta_u  = Eigen::VectorXf::Zero(TC);
    L.beta_m  = Eigen::VectorXf::Zero(TC);
    L.beta_v  = Eigen::VectorXf::Zero(TC);

    L.norm_hist       = Eigen::MatrixXf::Zero(TC, H);
    L.state_hist      = Eigen::MatrixXf::Zero(TC, H);
    L.ivar_hist       = Eigen::Matrix<float, 3, Eigen::Dynamic>::Zero(3, H);
    L.tanh_state_hist = Eigen::MatrixXf::Zero(C, H);
    L.last_cell_hist  = Eigen::MatrixXf::Zero(C, H);

    L.cell_state   = Eigen::VectorXf::Zero(C);
    L.state_error  = Eigen::VectorXf::Zero(C);
    L.stored_error = Eigen::VectorXf::Zero(C);
    L.gate_error   = Eigen::VectorXf::Zero(TC);

    // Xavier init for all three gate rows, same random sequence as original:
    //   forget, input, output per-row across columns except the very last
    //   (bias col). Forget-gate bias col set to 1.
    for (int row = 0; row < C; ++row) {
      for (int col = 0; col < static_cast<int>(total_cols_); ++col) {
        L.W(row + 0 * C, col) = low + get_uniform_random() * range;  // forget
        L.W(row + 1 * C, col) = low + get_uniform_random() * range;  // input
        L.W(row + 2 * C, col) = low + get_uniform_random() * range;  // output
      }
      L.W(row + 0 * C, total_cols_ - 1) = 1.0f;  // forget bias col = 1
    }
  }
}

inline void LstmFast::SetInput(const Eigen::VectorXf& input) {
  // head(input_size_) of each layer_input column = input.
  for (unsigned int l = 0; l < num_layers_; ++l) {
    layer_inputs_[l].col(epoch_).head(input_size_) = input.head(input_size_);
  }
}

// ---------------------------------------------------------------------------
// Forward pass for one layer. All three gates fused into a single GEMV.
// layer_in is the current epoch's input column (already wired in Predict).
// ---------------------------------------------------------------------------
inline void LstmFast::LayerForward(Layer& L, ConstVecRef layer_in,
                                  int input_symbol, int layer_idx) {
  const int C  = static_cast<int>(num_cells_);
  const int e  = static_cast<int>(epoch_);
  const int V  = static_cast<int>(output_size_);

  // Save previous cell state for this epoch.
  L.last_cell_hist.col(e) = L.cell_state;

  // Single fused GEMV producing pre-activations for all 3 gates (size 3C).
  // state_hist.col(e) is reused as scratch for the pre-activation, then
  // becomes the final post-activation output by the end.
  auto st = L.state_hist.col(e);
  st.noalias()  = L.W.col(input_symbol);
  st.noalias() += L.W.middleCols(V, layer_input_dim_) * layer_in;

  // Per-gate LN: normalize then scale+shift with gamma/beta.
  // norm_hist stores the normalized value pre gamma/beta; state_hist holds
  // post LN (which then gets activation applied in-place below).
  for (int g = 0; g < 3; ++g) {
    auto seg_state = st.segment(g * C, C);
    auto seg_norm  = L.norm_hist.col(e).segment(g * C, C);
    float inv_std  = 1.0f / std::sqrt(seg_state.squaredNorm() / C + 1e-5f);
    L.ivar_hist(g, e) = inv_std;
    seg_norm = seg_state * inv_std;
    auto seg_gamma = L.gamma.segment(g * C, C);
    auto seg_beta  = L.beta.segment(g * C, C);
    seg_state.array() = seg_norm.array() * seg_gamma.array() + seg_beta.array();
  }

  // In-place activations on contiguous 3C buffer:
  //   forget  [0..C)     sigmoid
  //   input   [C..2C)    tanh
  //   output  [2C..3C)   sigmoid
  auto f_seg = st.segment(0 * C, C);
  auto i_seg = st.segment(1 * C, C);
  auto o_seg = st.segment(2 * C, C);
  f_seg.array() = 1.0f / (1.0f + (-f_seg.array()).exp());
  i_seg.array() = i_seg.array().tanh();
  o_seg.array() = 1.0f / (1.0f + (-o_seg.array()).exp());

  // Cell state update: c = c * f + i * (1 - f) ; h = o * tanh(c)
  // (input_gate_state = 1 - f is not stored; recomputed in backward.)
  L.cell_state.array() = L.cell_state.array() * f_seg.array() +
                         i_seg.array() * (1.0f - f_seg.array());
  L.tanh_state_hist.col(e).array() = L.cell_state.array().tanh();

  const int h_off = layer_idx * C;
  hidden_.segment(h_off, C).array() =
      o_seg.array() * L.tanh_state_hist.col(e).array();
}

// ---------------------------------------------------------------------------
// Backward pass for one layer at a given epoch. Matches LstmLayer::BackwardPass
// numerically (modulo float-assoc reordering under -ffast-math).
// ---------------------------------------------------------------------------
inline void LstmFast::LayerBackward(Layer& L, ConstVecRef layer_in,
                                   int epoch, int layer_idx, int input_symbol,
                                   Eigen::VectorXf& hidden_error,
                                   int bptt_start) {
  const int C  = static_cast<int>(num_cells_);
  const int V  = static_cast<int>(output_size_);

  if (epoch == static_cast<int>(horizon_) - 1) {
    L.stored_error = hidden_error;
    L.state_error.setZero();
  } else {
    L.stored_error += hidden_error;
  }

  auto f_seg = L.state_hist.col(epoch).segment(0 * C, C);
  auto i_seg = L.state_hist.col(epoch).segment(1 * C, C);
  auto o_seg = L.state_hist.col(epoch).segment(2 * C, C);
  auto tanh_c = L.tanh_state_hist.col(epoch);
  auto last_c = L.last_cell_hist.col(epoch);

  // Output gate gradient
  auto go_err = L.gate_error.segment(2 * C, C);
  go_err.array() = tanh_c.array() * L.stored_error.array() *
                   o_seg.array() * (1.0f - o_seg.array());

  // State error accumulates
  L.state_error.array() += L.stored_error.array() * o_seg.array() *
                           (1.0f - tanh_c.array().square());

  // Input node gradient (derivative of tanh)
  auto gi_err = L.gate_error.segment(1 * C, C);
  gi_err.array() = L.state_error.array() * (1.0f - f_seg.array()) *
                   (1.0f - i_seg.array().square());

  // Forget gate gradient
  auto gf_err = L.gate_error.segment(0 * C, C);
  gf_err.array() = (last_c.array() - i_seg.array()) * L.state_error.array() *
                   f_seg.array() * (1.0f - f_seg.array());

  hidden_error.setZero();

  if (epoch > bptt_start) {
    L.state_error.array() *= f_seg.array();
    L.stored_error.setZero();
  } else {
    if (L.update_steps < UPDATE_LIMIT) {
      ++L.update_steps;
      L.beta1_power *= 0.025f;
      L.beta2_power *= 0.9999f;
    }
  }

  // LN backward (per gate, matches original):
  //   beta_u += err
  //   gamma_u += err * norm
  //   err = err * gamma * ivar
  //   err -= mean(err * norm) * norm   (mean over C)
  L.beta_u  += L.gate_error;
  L.gamma_u.array() += L.gate_error.array() * L.norm_hist.col(epoch).array();

  for (int g = 0; g < 3; ++g) {
    auto err  = L.gate_error.segment(g * C, C);
    auto nrm  = L.norm_hist.col(epoch).segment(g * C, C);
    auto gam  = L.gamma.segment(g * C, C);
    const float iv = L.ivar_hist(g, epoch);
    err.array() = err.array() * gam.array() * iv;
    float mean_en = (err.array() * nrm.array()).sum() / static_cast<float>(C);
    err.array() -= mean_en * nrm.array();
  }

  // Hidden-error propagation to previous layer (only if multi-layer).
  // W_prev block = columns [V + in + C .. V + in + 2C).
  if (layer_idx > 0) {
    const int prev_col = V + static_cast<int>(input_size_) + C;
    hidden_error.noalias() +=
        L.W.middleCols(prev_col, C).transpose() * L.gate_error;
  }

  // Recurrent error to previous timestep.
  if (epoch > bptt_start) {
    const int rec_col = V + static_cast<int>(input_size_);
    L.stored_error.noalias() +=
        L.W.middleCols(rec_col, C).transpose() * L.gate_error;
  }

  // Weight gradient accumulation (fused across gates):
  //   W_update[:, V:V+in_dim] += gate_error * layer_in.T  (one outer product)
  //   W_update[:, input_symbol] += gate_error             (one col add)
  L.W_update.middleCols(V, layer_input_dim_).noalias() +=
      L.gate_error * layer_in.transpose();
  L.W_update.col(input_symbol) += L.gate_error;

  if (epoch == bptt_start) {
    const float bc1   = 1.0f - L.beta1_power;
    const float bc2   = 1.0f - L.beta2_power;
    const float alpha = lstm_fast_detail::AdamAlpha(learning_rate_,
                                                    L.update_steps);
    lstm_fast_detail::AdamMat(L.W_update, L.W_m, L.W_v, L.W,
                              alpha, bc1, bc2);
    lstm_fast_detail::AdamVec(L.gamma_u, L.gamma_m, L.gamma_v, L.gamma,
                              alpha, bc1, bc2);
    lstm_fast_detail::AdamVec(L.beta_u,  L.beta_m,  L.beta_v,  L.beta,
                              alpha, bc1, bc2);
  }

  // Clip (state_error, stored_error, hidden_error).
  const float gc = gradient_clip_;
  L.state_error   = L.state_error.array().max(-gc).min(gc);
  L.stored_error  = L.stored_error.array().max(-gc).min(gc);
  hidden_error    = hidden_error.array().max(-gc).min(gc);
}

inline Eigen::VectorXf& LstmFast::Perceive(unsigned int input) {
  const int C = static_cast<int>(num_cells_);
  const int V = static_cast<int>(output_size_);
  const int H = static_cast<int>(horizon_);

  int last_epoch = static_cast<int>(epoch_) - 1;
  if (last_epoch == -1) last_epoch = H - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = static_cast<uint8_t>(input);

  // Compute the rank-1 errors for the last Predict (whose result is still in
  // output_current_) and stash them into errors_ring_ BEFORE BPTT runs.
  // This lets BPTT reconstruct `output_history_.col(epoch) -
  // onehot(input_history_[epoch])` as `errors_ring_.col((epoch+1) mod H) /
  // learning_rate_`, eliminating the per-bit 256-float store into a separate
  // output_history_ matrix. The BPTT rank-1 correction loop reads
  // errors_ring_.col(k) only for k>epoch (k >= 1), so writing col(epoch_)=0
  // early is safe.
  Eigen::VectorXf errors = output_current_;
  errors[input] -= 1.0f;
  errors *= learning_rate_;
  errors_ring_.col(epoch_) = errors;

  if (epoch_ == 0 && ++bptt_cycle_ >= bptt_period_) {
    bptt_cycle_ = 0;
    const int bptt_start = H - bptt_depth_;

    // Pre-zero the per-BPTT accumulators (hoisted out of the inner loop).
    for (int l = 0; l < static_cast<int>(num_layers_); ++l) {
      Layer& L = layers_[l];
      L.gamma_u.setZero();
      L.beta_u.setZero();
      L.W_update.setZero();
    }

    const float inv_lr = 1.0f / learning_rate_;

    for (int epoch = H - 1; epoch >= bptt_start; --epoch) {
      // errors_vec is shared by all layers at this epoch.
      // errors_ring_.col(k) stores (hist[k-1] - onehot(input_history_[k-1]))
      // * lr; so for timestep `epoch` we need column (epoch+1) mod H.
      int ring_idx = epoch + 1;
      if (ring_idx == H) ring_idx = 0;
      Eigen::VectorXf errors_vec = errors_ring_.col(ring_idx) * inv_lr;

      // Reconstruct `slot[epoch] * errors_vec` for the full hidden-range at
      // once, then each layer reads its own segment. See note in the header:
      //   slot[epoch] * v = M_cur * v + sum_{k>epoch} hidden_ring.col(k)
      //                                    * (errors_ring.col(k) · v)
      Eigen::VectorXf slot_times_err = output_layer_ * errors_vec;
      // Add sum of rank-1 corrections from ring positions (epoch+1 .. H-1).
      // (Attempted to collapse into two GEMVs in v5 but that regressed by ~4 %:
      // H=128 is too small to amortize GEMV dispatch cost, and the rewrite
      // needed a per-epoch `coeffs` temporary. See lstm-optimization.md.)
      for (int k = epoch + 1; k < H; ++k) {
        float coeff = errors_ring_.col(k).dot(errors_vec);
        slot_times_err.noalias() += coeff * hidden_ring_.col(k);
      }

      for (int layer = static_cast<int>(num_layers_) - 1; layer >= 0; --layer) {
        const int h_off = layer * C;

        hidden_error_ += slot_times_err.segment(h_off, C);

        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = H - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;

        LayerBackward(layers_[layer], layer_inputs_[layer].col(epoch),
                      epoch, layer, input_symbol, hidden_error_, bptt_start);
      }
    }
  }

  // Output-layer rank-1 update (single live matrix, no per-epoch snapshot).
  // `errors` and `errors_ring_.col(epoch_)` were already computed above,
  // before BPTT; reuse them here.
  hidden_ring_.col(epoch_) = hidden_;
  output_layer_.noalias() -= hidden_ * errors.transpose();

  return Predict(input);
}

inline Eigen::VectorXf& LstmFast::Predict(unsigned int input) {
  const int C = static_cast<int>(num_cells_);

  for (unsigned int l = 0; l < num_layers_; ++l) {
    // Wire recurrent block = previous epoch's hidden from this layer.
    layer_inputs_[l].col(epoch_).segment(input_size_, num_cells_) =
        hidden_.segment(l * num_cells_, num_cells_);

    LayerForward(layers_[l], layer_inputs_[l].col(epoch_),
                 static_cast<int>(input), static_cast<int>(l));

    // Wire the prev-layer block of the next layer with just-computed hidden.
    if (l + 1 < num_layers_) {
      layer_inputs_[l + 1].col(epoch_)
          .segment(num_cells_ + input_size_, num_cells_) =
              hidden_.segment(l * num_cells_, num_cells_);
    }
  }

  // Output softmax: logits = output_layer^T * hidden, stored into output_current_.
  output_current_.noalias() = output_layer_.transpose() * hidden_;
  float mx = output_current_.maxCoeff();
  output_current_.array() = (output_current_.array() - mx).exp();
  output_current_ /= output_current_.sum();

  // output_history_ is no longer persisted here; Perceive reconstructs past
  // per-epoch errors from errors_ring_ instead (see comment in Perceive).

  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = static_cast<int>(input);
  return output_current_;
}

#endif  // LSTM_FAST_HPP
