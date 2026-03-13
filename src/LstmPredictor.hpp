#ifndef LSTM_PREDICTOR_HPP
#define LSTM_PREDICTOR_HPP

#include "IPredictor.h"
#include "mixer/lstm.h"
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

/**
 * LstmPredictor - Isolated LSTM predictor for performance analysis and tuning
 *
 * Uses only the LSTM byte mixer without PPMd, FXCM, or any other models.
 * Feeds a uniform prior to the LSTM each byte (no external model input).
 * Converts byte-level probabilities to bit-level predictions using
 * the standard bot/mid/top binary decomposition (same as ByteModel).
 *
 * Usage:
 *   LstmPredictor predictor(200, 1, 128, 0.03, 10);
 *   float p = predictor.Predict();
 *   predictor.Perceive(bit);
 */
class LstmPredictor : public IPredictor {
public:
  LstmPredictor(unsigned int num_cells = 200, unsigned int num_layers = 1,
                int horizon = 128, float learning_rate = 0.03f,
                float gradient_clip = 10.0f, int bptt_depth = 0,
                int bptt_period = 1,
                const std::vector<bool> &vocab = std::vector<bool>(256, true))
      : bit_context_(1), top_(255), mid_(0), bot_(0), vocab_(vocab),
        probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)),
        num_cells_(num_cells), num_layers_(num_layers), horizon_(horizon),
        learning_rate_(learning_rate), gradient_clip_(gradient_clip) {

    // Count active vocabulary bytes
    vocab_size_ = 0;
    for (int i = 0; i < 256; ++i) {
      if (vocab_[i])
        ++vocab_size_;
    }

    // Build byte → vocab index map
    byte_map_ = Eigen::VectorXi::Zero(256);
    int offset = 0;
    for (int i = 0; i < 256; ++i) {
      byte_map_[i] = offset;
      if (vocab_[i])
        ++offset;
    }

    // Create LSTM: input = vocab_size (uniform prior), output = vocab_size
    lstm_ = std::make_unique<Lstm>(vocab_size_, vocab_size_, num_cells,
                                   num_layers, horizon, learning_rate,
                                   gradient_clip, bptt_depth, bptt_period);

    // Uniform input vector (no external model, just a constant prior)
    uniform_input_ = Eigen::VectorXf::Constant(vocab_size_, 1.0f / vocab_size_);

    std::cout << "LstmPredictor initialized:" << std::endl;
    std::cout << "  Cells: " << num_cells << ", Layers: " << num_layers
              << std::endl;
    std::cout << "  Horizon: " << horizon << ", LR: " << learning_rate
              << std::endl;
    std::cout << "  Gradient clip: " << gradient_clip << std::endl;
    std::cout << "  Vocab size: " << vocab_size_ << " bytes" << std::endl;
    if (bptt_depth > 0)
      std::cout << "  BPTT depth: " << bptt_depth << std::endl;
    if (bptt_period > 1)
      std::cout << "  BPTT period: " << bptt_period << std::endl;
  }

  /**
   * Predict probability of next bit being 1.
   * Uses bot/mid/top binary decomposition of byte probabilities.
   */
  float Predict() override {
    int   mid   = bot_ + ((top_ - bot_) / 2);
    float num   = probs_.segment(mid + 1, top_ - mid).sum();
    float denom = probs_.segment(bot_, mid + 1 - bot_).sum() + num;
    if (denom == 0)
      return 0.5f;
    return num / denom;
  }

  /**
   * Update with actual bit value.
   * On byte boundary, feed LSTM and get new byte probabilities.
   */
  void Perceive(int bit) override {
    // Update bit-level range (same as ByteModel::Perceive)
    mid_ = bot_ + ((top_ - bot_) / 2);
    if (bit) {
      bot_ = mid_ + 1;
    } else {
      top_ = mid_;
    }

    // Update bit context
    bit_context_ = (bit_context_ << 1) | bit;

    // Byte boundary: we have a complete byte
    if (bit_context_ >= 256) {
      unsigned int completed_byte = bit_context_ - 256;

      // Feed uniform input to LSTM and get output
      lstm_->SetInput(uniform_input_);
      const auto &output = lstm_->Perceive(byte_map_[completed_byte]);

      // Convert LSTM output back to 256-byte probability space
      int offset = 0;
      for (int i = 0; i < 256; ++i) {
        if (vocab_[i]) {
          probs_[i] = output[offset];
          ++offset;
        } else {
          probs_[i] = 0;
        }
      }

      // Reset for next byte
      top_         = 255;
      bot_         = 0;
      bit_context_ = 1;
    }
  }

  void Pretrain(int bit) override {}

  unsigned int GetNumCells() const { return num_cells_; }
  unsigned int GetNumLayers() const { return num_layers_; }
  int          GetHorizon() const { return horizon_; }

private:
  unsigned int       bit_context_;
  int                top_, mid_, bot_;
  std::vector<bool>  vocab_;
  Eigen::VectorXf    probs_;
  Eigen::VectorXi    byte_map_;
  unsigned int       vocab_size_;
  unsigned int       num_cells_;
  unsigned int       num_layers_;
  int                horizon_;
  float              learning_rate_;
  float              gradient_clip_;
  std::unique_ptr<Lstm> lstm_;
  Eigen::VectorXf    uniform_input_;
};

#endif // LSTM_PREDICTOR_HPP
