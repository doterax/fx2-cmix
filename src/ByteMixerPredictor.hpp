#ifndef BYTE_MIXER_PREDICTOR_HPP
#define BYTE_MIXER_PREDICTOR_HPP

#include "IPredictor.h"
#include "mixer/lstm.h"
#include "models/ppmd.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

// Isolated PPMd + LSTM predictor (ByteMixer without the full 461-model ensemble).
// Data flow: PPMd produces byte probs → LSTM refines them → bit-level decomposition.
class ByteMixerPredictor : public IPredictor {
public:
  ByteMixerPredictor(int ppmd_order = 25, int ppmd_mb = 1024,
                     unsigned int lstm_cells = 200, unsigned int lstm_layers = 1,
                     int horizon = 128, float learning_rate = 0.03f,
                     float gradient_clip = 10.0f,
                     int bptt_depth = 0, int bptt_period = 1,
                     const std::vector<bool> &vocab = std::vector<bool>(256, true))
      : bit_context_(1), top_(255), mid_(0), bot_(0), vocab_(vocab),
        probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)) {

    vocab_size_ = 0;
    for (int i = 0; i < 256; ++i) {
      if (vocab_[i]) ++vocab_size_;
    }

    byte_map_ = Eigen::VectorXi::Zero(256);
    int offset = 0;
    for (int i = 0; i < 256; ++i) {
      byte_map_[i] = offset;
      if (vocab_[i]) ++offset;
    }

    ppmd_ = std::make_unique<PPMD::PPMD>(ppmd_order, ppmd_mb, bit_context_, vocab_);

    lstm_ = std::make_unique<Lstm>(vocab_size_, vocab_size_, lstm_cells,
                                   lstm_layers, horizon, learning_rate,
                                   gradient_clip, bptt_depth, bptt_period);

    lstm_input_ = Eigen::VectorXf::Zero(vocab_size_);
  }

  float Predict() override {
    // PPMd must still be called each bit to keep its internal state in sync
    ppmd_->Predict();

    // LSTM byte probs already incorporate PPMd input — use them directly
    int   mid   = bot_ + ((top_ - bot_) / 2);
    float num   = probs_.segment(mid + 1, top_ - mid).sum();
    float denom = probs_.segment(bot_, mid + 1 - bot_).sum() + num;
    if (denom == 0) return 0.5f;
    return num / denom;
  }

  void Perceive(int bit) override {
    ppmd_->Perceive(bit);

    mid_ = bot_ + ((top_ - bot_) / 2);
    if (bit) { bot_ = mid_ + 1; } else { top_ = mid_; }

    bit_context_ = (bit_context_ << 1) | bit;

    if (bit_context_ >= 256) {
      unsigned int completed_byte = bit_context_ - 256;
      bit_context_ = completed_byte;

      // PPMd byte update
      ppmd_->ByteUpdate();

      // Get PPMd byte-level probabilities and feed to LSTM
      const Eigen::VectorXf &ppmd_byte_probs = ppmd_->BytePredict();
      int off = 0;
      for (int i = 0; i < 256; ++i) {
        if (vocab_[i]) {
          lstm_input_[off] = ppmd_byte_probs[i];
          ++off;
        }
      }

      lstm_->SetInput(lstm_input_);
      const auto &output = lstm_->Perceive(byte_map_[completed_byte]);

      off = 0;
      for (int i = 0; i < 256; ++i) {
        if (vocab_[i]) { probs_[i] = output[off]; ++off; }
        else           { probs_[i] = 0; }
      }

      top_ = 255; bot_ = 0; bit_context_ = 1;
    }
  }

  void Pretrain(int bit) override {}

private:
  unsigned int bit_context_;
  int top_, mid_, bot_;
  std::vector<bool> vocab_;
  Eigen::VectorXf probs_, lstm_input_;
  Eigen::VectorXi byte_map_;
  unsigned int vocab_size_;
  std::unique_ptr<PPMD::PPMD> ppmd_;
  std::unique_ptr<Lstm> lstm_;
};

#endif
