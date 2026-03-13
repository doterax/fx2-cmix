#ifndef BYTE_MIXER_PREDICTOR_HPP
#define BYTE_MIXER_PREDICTOR_HPP

#include "IPredictor.h"
#include "mixer/lstm.h"
#include "models/ppmd.h"
#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

// Micro-mixer: online logistic regression over 2 stretched inputs.
// Learns optimal PPMd vs LSTM blending from actual bit outcomes.
class MicroMixer {
public:
  MicroMixer(float lr = 0.07f) : w_{0.0f, 0.0f}, lr_(lr) {}

  static float Stretch(float p) {
    p = std::max(0.0001f, std::min(0.9999f, p));
    return std::log(p / (1.0f - p));
  }
  static float Squash(float x) { return 1.0f / (1.0f + std::exp(-x)); }

  float Mix(float p0, float p1) {
    s0_ = Stretch(p0);
    s1_ = Stretch(p1);
    return Squash(w_[0] * s0_ + w_[1] * s1_);
  }

  void Update(int bit) {
    float p = Squash(w_[0] * s0_ + w_[1] * s1_);
    float err = (static_cast<float>(bit) - p) * lr_;
    w_[0] += err * s0_;
    w_[1] += err * s1_;
  }

private:
  float w_[2];
  float s0_, s1_;
  float lr_;
};

// Classify a byte into one of 5 classes for mixer context.
// 0=lowercase, 1=uppercase, 2=digit, 3=space/newline, 4=other
inline int ByteClass(unsigned int c) {
  if (c >= 'a' && c <= 'z') return 0;
  if (c >= 'A' && c <= 'Z') return 1;
  if (c >= '0' && c <= '9') return 2;
  if (c == ' ' || c == '\n' || c == '\r' || c == '\t') return 3;
  return 4;
}

static constexpr int kNumByteClasses = 5;
static constexpr int kNumBitPositions = 8;
static constexpr int kNumMixers = kNumByteClasses * kNumBitPositions; // 40

// Isolated PPMd + LSTM predictor with adaptive micro-mixer.
// Data flow: PPMd produces byte probs → LSTM refines them → micro-mixer
// blends PPMd bit prediction with LSTM bit prediction, learning optimal
// weights online from actual bit outcomes.
// v5: 40 mixers indexed by (previous byte class × bit position).
class ByteMixerPredictor : public IPredictor {
public:
  ByteMixerPredictor(int ppmd_order = 25, int ppmd_mb = 1024,
                     unsigned int lstm_cells = 200, unsigned int lstm_layers = 1,
                     int horizon = 128, float learning_rate = 0.03f,
                     float gradient_clip = 10.0f,
                     int bptt_depth = 0, int bptt_period = 1,
                     const std::vector<bool> &vocab = std::vector<bool>(256, true))
      : bit_context_(1), top_(255), mid_(0), bot_(0), prev_byte_class_(4),
        vocab_(vocab),
        probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)),
        last_mix_p_(0.5f) {

    // 40 micro-mixers: 5 byte classes × 8 bit positions
    for (int i = 0; i < kNumMixers; ++i) {
      mixers_[i] = MicroMixer(0.07f);
    }

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
    // Get PPMd bit prediction
    const Eigen::VectorXf &ppmd_bit = ppmd_->Predict();
    float ppmd_p = ppmd_bit[0];

    // Get LSTM bit prediction from byte-level probs
    int   mid   = bot_ + ((top_ - bot_) / 2);
    float num   = probs_.segment(mid + 1, top_ - mid).sum();
    float denom = probs_.segment(bot_, mid + 1 - bot_).sum() + num;
    float lstm_p = (denom == 0) ? 0.5f : num / denom;

    // Select micro-mixer by (prev byte class × bit position)
    int bit_pos = 0;
    unsigned int bc = bit_context_ >> 1;
    while (bc > 0) { ++bit_pos; bc >>= 1; }
    int mixer_idx = prev_byte_class_ * kNumBitPositions + bit_pos;

    last_mix_p_ = mixers_[mixer_idx].Mix(ppmd_p, lstm_p);
    return last_mix_p_;
  }

  void Perceive(int bit) override {
    // Update the (prev byte class × bit position) mixer that made the last prediction
    int bit_pos = 0;
    unsigned int bc = bit_context_ >> 1;
    while (bc > 0) { ++bit_pos; bc >>= 1; }
    int mixer_idx = prev_byte_class_ * kNumBitPositions + bit_pos;
    mixers_[mixer_idx].Update(bit);

    ppmd_->Perceive(bit);

    mid_ = bot_ + ((top_ - bot_) / 2);
    if (bit) { bot_ = mid_ + 1; } else { top_ = mid_; }

    bit_context_ = (bit_context_ << 1) | bit;

    if (bit_context_ >= 256) {
      unsigned int completed_byte = bit_context_ - 256;
      bit_context_ = completed_byte;

      ppmd_->ByteUpdate();

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

      prev_byte_class_ = ByteClass(completed_byte);
      top_ = 255; bot_ = 0; bit_context_ = 1;
    }
  }

  void Pretrain(int bit) override {}

private:
  unsigned int bit_context_;
  int top_, mid_, bot_;
  int prev_byte_class_;
  std::vector<bool> vocab_;
  Eigen::VectorXf probs_, lstm_input_;
  Eigen::VectorXi byte_map_;
  unsigned int vocab_size_;
  std::unique_ptr<PPMD::PPMD> ppmd_;
  std::unique_ptr<Lstm> lstm_;
  MicroMixer mixers_[kNumMixers];
  float last_mix_p_;
};

#endif
