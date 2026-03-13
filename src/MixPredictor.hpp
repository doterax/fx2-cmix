#ifndef MIX_PREDICTOR_HPP
#define MIX_PREDICTOR_HPP

#include "IPredictor.h"
#include "contexts/bracket-context.h"
#include "mixer/lstm.h"
#include "mixer/sse.h"
#include "models/bracket.h"
#include "models/ppmd.h"
#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

// N-input micro-mixer: online logistic regression over N stretched inputs.
class NMicroMixer {
public:
  NMicroMixer() : n_(0), lr_(0.07f) {}
  explicit NMicroMixer(int n, float lr = 0.07f)
      : n_(n), w_(n, 0.0f), s_(n, 0.0f), lr_(lr) {}

  static float Stretch(float p) {
    p = std::max(0.0001f, std::min(0.9999f, p));
    return std::log(p / (1.0f - p));
  }
  static float Squash(float x) { return 1.0f / (1.0f + std::exp(-x)); }

  float Mix(const float* probs, int n) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      s_[i] = Stretch(probs[i]);
      sum += w_[i] * s_[i];
    }
    return Squash(sum);
  }

  void Update(int bit) {
    float sum = 0;
    for (int i = 0; i < n_; ++i) sum += w_[i] * s_[i];
    float p = Squash(sum);
    float err = (static_cast<float>(bit) - p) * lr_;
    for (int i = 0; i < n_; ++i) {
      w_[i] += err * s_[i];
    }
  }

private:
  int n_;
  std::vector<float> w_;
  std::vector<float> s_;
  float lr_;
};

// Classify a byte into one of 5 classes for mixer context.
// 0=lowercase, 1=uppercase, 2=digit, 3=space/newline, 4=other
inline int MixByteClass(unsigned int c) {
  if (c >= 'a' && c <= 'z') return 0;
  if (c >= 'A' && c <= 'Z') return 1;
  if (c >= '0' && c <= '9') return 2;
  if (c == ' ' || c == '\n' || c == '\r' || c == '\t') return 3;
  return 4;
}

static constexpr int kMixNumByteClasses = 5;
static constexpr int kMixNumBitPositions = 8;
static constexpr int kMixNumBracketStates = 2; // 0=outside, 1=inside bracket
static constexpr int kMixNumContexts =
    kMixNumByteClasses * kMixNumBitPositions * kMixNumBracketStates; // 80

// Extensible multi-model predictor.
// Combines PPMd+LSTM (via internal ByteMixer logic) + Bracket + future models.
// Each model produces a bit probability; an N-input context-indexed micro-mixer
// blends them.
class MixPredictor : public IPredictor {
public:
  MixPredictor(int ppmd_order = 25, int ppmd_mb = 1024,
               unsigned int lstm_cells = 200, unsigned int lstm_layers = 1,
               int horizon = 128, float learning_rate = 0.03f,
               float gradient_clip = 10.0f,
               int bptt_depth = 0, int bptt_period = 1,
               const std::vector<bool> &vocab = std::vector<bool>(256, true))
      : bit_context_(1), top_(255), mid_(0), bot_(0), prev_byte_class_(4),
        bracket_state_(0), vocab_(vocab),
        probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)) {

    // --- Sub-model count: PPMd bit + LSTM bit + Bracket bit = 3 inputs ---
    num_inputs_ = 3;

    // 80 N-input micro-mixers: 5 byte classes x 8 bit positions x 2 bracket states
    mixers_.reserve(kMixNumContexts);
    for (int i = 0; i < kMixNumContexts; ++i) {
      mixers_.emplace_back(num_inputs_, 0.07f);
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

    // PPMd
    ppmd_ = std::make_unique<PPMD::PPMD>(ppmd_order, ppmd_mb, bit_context_, vocab_);

    // LSTM
    lstm_ = std::make_unique<Lstm>(vocab_size_, vocab_size_, lstm_cells,
                                   lstm_layers, horizon, learning_rate,
                                   gradient_clip, bptt_depth, bptt_period);
    lstm_input_ = Eigen::VectorXf::Zero(vocab_size_);

    // Bracket model — shares bit_context_ reference
    bracket_ = std::make_unique<Bracket>(bit_context_, 200, 10, 100000, vocab_);

    // Bracket context — tracks nesting state for mixer context selection
    bracket_ctx_ = std::make_unique<BracketContext>(bit_context_, 200, 10);
  }

  float Predict() override {
    // --- Input 0: PPMd bit prediction ---
    const Eigen::VectorXf &ppmd_bit = ppmd_->Predict();
    float ppmd_p = ppmd_bit[0];

    // --- Input 1: LSTM bit prediction (from byte-level probs) ---
    int   mid   = bot_ + ((top_ - bot_) / 2);
    float num   = probs_.segment(mid + 1, top_ - mid).sum();
    float denom = probs_.segment(bot_, mid + 1 - bot_).sum() + num;
    float lstm_p = (denom == 0) ? 0.5f : num / denom;

    // --- Input 2: Bracket bit prediction ---
    float bracket_p = bracket_->Predict()[0];

    // --- Gather all sub-model probabilities ---
    float inputs[3] = { ppmd_p, lstm_p, bracket_p };

    // Select mixer by (prev byte class x bit position x bracket state)
    int bit_pos = 0;
    unsigned int bc = bit_context_ >> 1;
    while (bc > 0) { ++bit_pos; bc >>= 1; }
    int mixer_idx = (prev_byte_class_ * kMixNumBitPositions + bit_pos)
                    * kMixNumBracketStates + bracket_state_;

    float mix_p = mixers_[mixer_idx].Mix(inputs, num_inputs_);
    return sse_.Predict(mix_p);
  }

  void Perceive(int bit) override {
    // Update the context mixer
    int bit_pos = 0;
    unsigned int bc = bit_context_ >> 1;
    while (bc > 0) { ++bit_pos; bc >>= 1; }
    int mixer_idx = (prev_byte_class_ * kMixNumBitPositions + bit_pos)
                    * kMixNumBracketStates + bracket_state_;
    mixers_[mixer_idx].Update(bit);

    // Update SSE
    sse_.Perceive(bit);

    // Update sub-models
    ppmd_->Perceive(bit);
    bracket_->Perceive(bit);

    // Update LSTM byte-level binary search
    mid_ = bot_ + ((top_ - bot_) / 2);
    if (bit) { bot_ = mid_ + 1; } else { top_ = mid_; }

    bit_context_ = (bit_context_ << 1) | bit;

    if (bit_context_ >= 256) {
      unsigned int completed_byte = bit_context_ - 256;
      bit_context_ = completed_byte;

      // PPMd byte-level update
      ppmd_->ByteUpdate();

      // Bracket byte-level update (reads bit_context_ which is now the byte value)
      bracket_->ByteUpdate();

      // Update bracket context and derive binary state (inside/outside)
      bracket_ctx_->Update();
      bracket_state_ = (bracket_ctx_->GetContext() != 0) ? 1 : 0;

      // Feed PPMd byte probs to LSTM as input
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

      prev_byte_class_ = MixByteClass(completed_byte);
      top_ = 255; bot_ = 0; bit_context_ = 1;
    }
  }

  void Pretrain(int bit) override {}

private:
  unsigned int bit_context_;
  int top_, mid_, bot_;
  int prev_byte_class_;
  int bracket_state_;
  int num_inputs_;
  std::vector<bool> vocab_;
  Eigen::VectorXf probs_, lstm_input_;
  Eigen::VectorXi byte_map_;
  unsigned int vocab_size_;
  std::unique_ptr<PPMD::PPMD> ppmd_;
  std::unique_ptr<Lstm> lstm_;
  std::unique_ptr<Bracket> bracket_;
  std::unique_ptr<BracketContext> bracket_ctx_;
  std::vector<NMicroMixer> mixers_;
  SSE sse_;
};

#endif
