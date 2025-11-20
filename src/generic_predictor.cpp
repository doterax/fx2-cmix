#include "generic_predictor.h"
#include <Eigen/Core>
#include <iostream>

static inline float clampProb(float p) {
  if (p < 1e-9f) return 1e-9f;
  if (p > 1.0f - 1e-9f) return 1.0f - 1e-9f;
  return p;
}

std::unique_ptr<GenericPredictor> GenericPredictor::Builder::build() {
  return std::unique_ptr<GenericPredictor>(new GenericPredictor(std::move(sources_), std::move(mixers_), num_layers_, sigmoid_size_, lstm_spec_));
}

GenericPredictor::GenericPredictor(std::vector<Source> sources, std::vector<MixerSpec> mixers,
                                   int num_layers, int sigmoid_size, LstmSpec lstm_spec)
  : sources_(std::move(sources)), mixer_specs_(std::move(mixers)), sigmoid_(sigmoid_size), lstm_spec_(lstm_spec) {
  if (num_layers < 1) num_layers = 1;
  if (num_layers > 2) num_layers = 2; // current implementation supports up to 2
  layer_inputs_.reserve(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    layer_inputs_.emplace_back(sigmoid_, 1.0e-4f);
  }

  // Layer 0 holds source predictions
  layer_inputs_[0].SetNumModels((int)sources_.size());

  // Prepare mixers per spec
  for (auto &m : mixer_specs_) {
    if (m.layer == 0) {
      unsigned int extraSize = (unsigned int)mixers_layer0_.size();
      mixers_layer0_.emplace_back(layer_inputs_[0].Inputs(), layer_inputs_[0].ExtraInputs(), *m.contextPtr, m.learning_rate, extraSize);
    } else if (layer_inputs_.size() > 1) {
      // Layer-1 aggregator uses only stretched inputs; no extra inputs needed
      mixers_layer1_.emplace_back(layer_inputs_[0].Inputs(), layer_inputs_[0].ExtraInputs(), *m.contextPtr, m.learning_rate, (unsigned int)0);
    }
  }

  if (layer_inputs_.size() > 1) {
    layer_inputs_[1].SetNumModels((int)mixers_layer0_.size());
    layer_inputs_[0].SetExtraInputSize(mixers_layer0_.size());
  }

  if (lstm_spec_.enabled()) {
    // output_size 2 => binary probability; input_size = sources_.size()
    lstm_.reset(new Lstm((unsigned int)sources_.size(), 2, lstm_spec_.num_cells,
                         lstm_spec_.num_layers, lstm_spec_.horizon,
                         lstm_spec_.learning_rate, lstm_spec_.gradient_clip));
  }
}

float GenericPredictor::Predict() {
  // Gather source probabilities
  int idx = 0;
  for (auto &s : sources_) {
    float p = clampProb(s.predictFn());
    layer_inputs_[0].SetInput(idx++, p);
  }

  // First layer mixers produce extra inputs
  for (size_t i = 0; i < mixers_layer0_.size(); ++i) {
    float mp = mixers_layer0_[i].Mix();
    if (layer_inputs_.size() > 1) {
      layer_inputs_[0].SetExtraInput(i, mp);
      layer_inputs_[1].SetStretchedInput((int)i, mp);
    }
  }

  float blended = 0.5f;
  if (!mixers_layer1_.empty()) {
    blended = mixers_layer1_[0].Mix();
    blended = Sigmoid::Logistic(blended);
    blended = sse_.Predict(blended);
  } else if (!mixers_layer0_.empty()) {
    // Fallback: average layer0 mixer outputs
    float sum = 0.0f;
    for (auto &m : mixers_layer0_) sum += Sigmoid::Logistic(m.Mix());
    blended = sum / mixers_layer0_.size();
  } else {
    // Fallback: average of sources
    float sum = 0.0f;
    for (auto &s : sources_) sum += s.predictFn();
    if (!sources_.empty()) blended = sum / sources_.size();
  }

  if (lstm_) {
    Eigen::VectorXf inputVec = Eigen::VectorXf::Zero((int)sources_.size());
    for (int i = 0; i < inputVec.size(); ++i) inputVec[i] = layer_inputs_[0].Inputs()[i];
    lstm_->SetInput(inputVec);
    auto &out = lstm_->Predict(0); // dummy index
    if (out.size() > 1) blended = out[1]; else blended = out[0];
  }

  return clampProb(blended);
}

void GenericPredictor::Perceive(int bit) {
  for (auto &s : sources_) {
    if (s.perceiveFn) s.perceiveFn(bit);
  }
  for (auto &m : mixers_layer0_) m.Perceive(bit);
  for (auto &m : mixers_layer1_) m.Perceive(bit);
  sse_.Perceive(bit);
  if (lstm_) lstm_->Perceive(0); // propagate learning with dummy symbol
}

void GenericPredictor::Pretrain(int bit) {
  (void)Predict();
  Perceive(bit);
}
