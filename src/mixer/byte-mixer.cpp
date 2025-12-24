#include "byte-mixer.h"

ByteMixer::ByteMixer(unsigned int num_models, const unsigned int &bit_context,
                     const std::vector<bool> &vocab, unsigned int vocab_size,
                     NMixer *lstm)
    : ByteModel(vocab), lstm_(lstm), byte_(bit_context),
      byte_map_(Eigen::VectorXi::Zero(256)),
      inputs_(Eigen::VectorXf::Zero(vocab_size)), num_models_(num_models),
      vocab_size_(vocab_size), ex(0), offset_(0) {
  for (int i = 0; i < 256; ++i) {
    byte_map_[i] = offset_;
    if (vocab_[i])
      ++offset_;
  }
  offset_ = 0;
}

void ByteMixer::SetInput(int index, float val) {
  if (!vocab_[index])
    return;
  inputs_[offset_] += val;
  ++offset_;
  if (offset_ == vocab_size_)
    offset_ = 0;
}

void ByteMixer::ByteUpdate() {
  inputs_ *= 2.0 / num_models_;
  lstm_->SetInput(inputs_);
  inputs_.setZero();
  const auto &output = lstm_->Perceive(byte_map_[byte_]);
  offset_            = 0;
  for (int i = 0; i < 256; ++i) {
    if (vocab_[i]) {
      probs_[i] = output[offset_];
      ++offset_;
    } else {
      probs_[i] = 0;
    }
  }

  offset_ = 0;
  ByteModel::ByteUpdate();
}

void ByteMixer::UpdateEx() {
  // Calculate ex - index with maximum probability in current bot_/top_ range
  ex                 = bot_;
  float max_prob_val = probs_[bot_];
  for (int i = bot_ + 1; i <= top_; ++i) {
    if (vocab_[i]) {
      if (probs_[i] > max_prob_val) {
        max_prob_val = probs_[i];
        ex           = i;
      }
    }
  }
}
