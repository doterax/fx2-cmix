#ifndef LSTM_FAST_H
#define LSTM_FAST_H

#include <Eigen/Core>
#include <vector>
#include <memory>

#include "NMixer.h"

// Drop-in replacement for `Lstm` from lstm.h with the same public API.
// Same numerical algorithm, same LN, same Adam schedule. Differences:
//   * the three LSTM gates are fused into one weight matrix per layer
//   * all per-epoch history is stored as column-major Eigen::MatrixXf
//     (contiguous per-timestep slices instead of std::vector<VectorXf>)
//   * a single output-layer snapshot is kept (no per-epoch copy)
//   * activations and LN fused inside the forward/backward passes
// See lstm-optimization.md for details.
class LstmFast : public NMixer {
 public:
  LstmFast(unsigned int input_size, unsigned int output_size,
          unsigned int num_cells, unsigned int num_layers, int horizon,
          float learning_rate, float gradient_clip,
          int bptt_depth = 0, int bptt_period = 1);
  ~LstmFast() override = default;

  Eigen::VectorXf& Perceive(unsigned int input) override;
  void             SetInput(const Eigen::VectorXf& input) override;

 private:
  struct Layer {
    // Fused gate weights W in R^{3C x total_cols}. Row-stripe layout:
    //   rows [0..C)      -> forget gate
    //   rows [C..2C)     -> input node
    //   rows [2C..3C)    -> output gate
    // Columns (same as original Lstm's layout):
    //   [0..V)                          -> embedding (looked up by input symbol)
    //   [V..V+in+2C+1)                  -> multiplied by layer_input vector
    //     sub-block aux  : [V            ..  V+in)
    //     sub-block recurrent : [V+in     ..  V+in+C)
    //     sub-block prev-layer: [V+in+C   ..  V+in+2C)
    //     sub-block bias col  : [V+in+2C  ..  V+in+2C+1)
    Eigen::MatrixXf W, W_update, W_m, W_v;       // (3C, total_cols)

    // LN parameters, stored contiguously as (3C) vectors.
    Eigen::VectorXf gamma, beta;                 // (3C)
    Eigen::VectorXf gamma_u, gamma_m, gamma_v;   // (3C)
    Eigen::VectorXf beta_u,  beta_m,  beta_v;    // (3C)

    // Per-epoch history (column per timestep, col-major so each column
    // is contiguous in memory).
    Eigen::MatrixXf norm_hist;         // (3C, H)  post-LN-normalize (pre gamma/beta)
    Eigen::MatrixXf state_hist;        // (3C, H)  post-activation (F|I|O)
    Eigen::Matrix<float, 3, Eigen::Dynamic> ivar_hist;  // (3, H) per-gate inv_std
    Eigen::MatrixXf tanh_state_hist;   // (C, H)
    Eigen::MatrixXf last_cell_hist;    // (C, H)

    Eigen::VectorXf cell_state;        // (C)  current c_t
    Eigen::VectorXf state_error;       // (C)
    Eigen::VectorXf stored_error;      // (C)

    // Scratch (size 3C) used inside Perceive.
    Eigen::VectorXf gate_error;        // (3C)

    // Adam bookkeeping (mirrors LstmLayer's private state).
    unsigned long long update_steps = 0;
    float              beta1_power  = 1.0f;
    float              beta2_power  = 1.0f;
  };

  Eigen::VectorXf& Predict(unsigned int input);

  // One forward/backward pass through a single layer (inlined, no extra class).
  // Ref<const VectorXf> accepts a contiguous column of a col-major matrix
  // without copying.
  using ConstVecRef = Eigen::Ref<const Eigen::VectorXf>;
  void LayerForward(Layer& L, ConstVecRef layer_in,
                    int input_symbol, int layer_idx);
  void LayerBackward(Layer& L, ConstVecRef layer_in,
                     int epoch, int layer_idx, int input_symbol,
                     Eigen::VectorXf& hidden_error, int bptt_start);

  std::vector<Layer>              layers_;
  std::vector<Eigen::MatrixXf>    layer_inputs_;   // [num_layers] (dim, H)
  std::vector<uint8_t>            input_history_;  // (H)

  Eigen::VectorXf hidden_, hidden_error_;
  Eigen::MatrixXf output_layer_;                   // (num_layers*C+1, output_size)
  Eigen::VectorXf output_current_;                 // (output_size)

  // Ring buffer of rank-1 updates applied to `output_layer_`.
  // At BPTT time we reconstruct the per-epoch snapshot that the original
  // `Lstm` kept as a separate matrix per epoch:
  //   slot[e] = output_layer_current + sum_{k=e+1..H-1} hidden_ring.col(k)
  //                                       * errors_ring.col(k).transpose()
  // so `slot[e].block(h_off, 0, C, V) * errors_vec` becomes
  //   output_layer_.block(...) * errors_vec
  //   + sum_{k>e} hidden_ring.col(k).segment(h_off, C) * (errors_ring.col(k) · errors_vec)
  // This restores bit-equivalent BPTT with the original without keeping
  // H full (C+1)xV matrices around.
  Eigen::MatrixXf hidden_ring_;                    // (num_layers*C+1, H)
  Eigen::MatrixXf errors_ring_;                    // (output_size, H)

  float        learning_rate_;
  float        gradient_clip_;
  unsigned int num_cells_;
  unsigned int input_size_;
  unsigned int output_size_;
  unsigned int num_layers_;
  unsigned int horizon_;
  unsigned int epoch_      = 0;
  int          bptt_depth_;
  int          bptt_period_;
  int          bptt_cycle_ = 0;
  int          last_input_ = -1;
  unsigned int layer_input_dim_;  // input_size + 2*num_cells + 1
  unsigned int total_cols_;       // output_size + layer_input_dim_
};

#include "lstm_fast.hpp"
#endif
