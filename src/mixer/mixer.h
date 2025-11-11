#ifndef MIXER_H
#define MIXER_H

#include <vector>
#include <Eigen/Core>
#include "../ds/emhash_map.hpp"
#include <memory>

struct ContextData {
  ContextData(unsigned long long input_size,
      unsigned long long extra_input_size) : /*steps(0), */weights(Eigen::VectorXf::Zero(input_size)),
      extra_weights(Eigen::VectorXf::Zero(extra_input_size)) {};
  ContextData& operator=(const ContextData&) = default;
  ContextData(const ContextData&) = default;
  //unsigned long long steps;
  Eigen::VectorXf weights, extra_weights;
  
};

class Mixer {
 public:
  Mixer(const Eigen::VectorXf& inputs,
      const Eigen::VectorXf& extra_inputs, const unsigned long long& context,
      float learning_rate, unsigned int extra_input_size);
  float Mix();
  void Perceive(int bit);

 private:
  ContextData* GetContextData();
  const Eigen::VectorXf& inputs_;
  const Eigen::VectorXf& extra_inputs_vec_;
  // Eigen::VectorXf extra_inputs_;
  uint16_t extra_inputs_size_;
  float p_, learning_rate_;
  const unsigned long long& context_;
  unsigned long long /*max_steps_,*/ steps_;
  emhash6::HashMap<unsigned int, ContextData> context_map_;
  ContextData context_base_;
};

#endif

