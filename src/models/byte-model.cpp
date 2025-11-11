#include "byte-model.h"

#include <numeric>

ByteModel::ByteModel(const std::vector<bool>& vocab) : ex(0),top_(255), mid_(0),
    bot_(0),  vocab_(vocab), probs_(Eigen::VectorXf::Constant(256, 1.0 / 256)) {}

 Eigen::VectorXf& ByteModel::Predict()  {
  auto mid = bot_ + ((top_ - bot_) / 2);
  float num = probs_.segment(mid + 1, top_ - mid).sum();
  float denom = probs_.segment(bot_, mid + 1 - bot_).sum() + num;
  ex = bot_;
    float max_prob_val = probs_[bot_];
    for (int i = bot_ + 1; i <= top_; i++) {
      if (probs_[i] > max_prob_val) {
        max_prob_val = probs_[i];
        ex = i;
      }
    }
  if (denom == 0) outputs_[0] = 0.5;
  else outputs_[0] = num / denom;
  return outputs_;
}

const Eigen::VectorXf& ByteModel::BytePredict() {
  return probs_;
}

void ByteModel::Perceive(int bit) {
  mid_ = bot_ + ((top_ - bot_) / 2);
  if (bit) {
    bot_ = mid_ + 1;
  } else {
    top_ = mid_;
  }
}

void ByteModel::ByteUpdate() {
  top_ = 255;
  bot_ = 0;
  for (int i = 0; i < 256; ++i) {
    if (!vocab_[i]) probs_[i] = 0;
  }
}

