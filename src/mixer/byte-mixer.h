#ifndef BYTE_MIXER_H
#define BYTE_MIXER_H

#include <vector>
#include <memory>
#include <Eigen/Core>

#include "../models/byte-model.h"
#include "NMixer.h"

class ByteMixer : public ByteModel {
 public:
  ByteMixer(unsigned int num_models, const unsigned int& bit_context,
      const std::vector<bool>& vocab, unsigned int vocab_size, NMixer* lstm);
  void SetInput(int index, float val);
  void ByteUpdate();

 private:
  std::unique_ptr<NMixer> lstm_;
  const unsigned int& byte_;
  Eigen::VectorXi byte_map_;
  Eigen::VectorXf inputs_;
  unsigned int num_models_, vocab_size_, offset_;
};

#endif
