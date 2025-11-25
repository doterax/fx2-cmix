#ifndef GENERIC_PREDICTOR_H
#define GENERIC_PREDICTOR_H

#include "IPredictor.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/sse.h"
#include "mixer/sigmoid.h"
#include "mixer/lstm.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Generic, model-agnostic predictor mixer with builder pattern.
// Sources provide raw probability estimates; mixers learn context-based weights.
// Optional LSTM layer can further refine blended probability.
class GenericPredictor : public IPredictor {
public:
  struct Source {
    std::string name;
    std::function<float()> predictFn;      // returns probability in (0,1)
    std::function<void(int)> perceiveFn;   // feedback bit
  };
  struct MixerSpec {
    const unsigned long long* contextPtr;  // external context value reference
    float learning_rate;
    int layer; // 0 or 1
  };
  struct LstmSpec {
    unsigned int num_cells = 0;
    unsigned int num_layers = 0;
    int horizon = 0;
    float learning_rate = 0.f;
    float gradient_clip = 0.f;
    bool enabled() const { return num_cells > 0 && num_layers > 0 && horizon > 0; }
  };

  class Builder {
  public:
    Builder& addSource(const std::string& name,
                       std::function<float()> predictFn,
                       std::function<void(int)> perceiveFn) {
      sources_.push_back({name, std::move(predictFn), std::move(perceiveFn)});
      return *this;
    }
    Builder& addMixer(const unsigned long long& context, float lr, int layer = 0) {
      mixers_.push_back({&context, lr, layer});
      return *this;
    }
    Builder& setLayers(int layers) {
      num_layers_ = layers; return *this;
    }
    Builder& enableLstm(unsigned int num_cells, unsigned int num_layers, int horizon,
                        float learning_rate, float gradient_clip) {
      lstm_spec_ = {num_cells, num_layers, horizon, learning_rate, gradient_clip};
      return *this;
    }
    Builder& setSigmoidTableSize(int size) { sigmoid_size_ = size; return *this; }
    Builder& enableSourceWeighting(float alpha = 0.001f, float min_w = 0.05f,
                                   float max_w = 1.5f, bool duplicate_input = true) {
      weighting_enabled_ = true;
      weight_alpha_ = alpha;
      min_weight_ = min_w;
      max_weight_ = max_w;
      duplicate_weighted_input_ = duplicate_input;
      return *this;
    }
    std::unique_ptr<GenericPredictor> build();
  private:
    std::vector<Source> sources_;
    std::vector<MixerSpec> mixers_;
    int num_layers_ = 2;
    int sigmoid_size_ = 100001;
    LstmSpec lstm_spec_{};
    // Source weighting config
    bool weighting_enabled_ = true;
    bool duplicate_weighted_input_ = true; // if true feed both raw and weighted
    float weight_alpha_ = 0.001f;
    float min_weight_ = 0.05f;
    float max_weight_ = 1.5f;
  };

  float Predict() override;
  void Perceive(int bit) override;
  void Pretrain(int bit) override; // no-op pretrain pass (calls Predict then Perceive)

  const std::vector<Source>& sources() const { return sources_; }

private:
  GenericPredictor(std::vector<Source> sources, std::vector<MixerSpec> mixers,
                   int num_layers, int sigmoid_size, LstmSpec lstm_spec,
                   bool weighting_enabled, bool duplicate_weighted_input,
                   float weight_alpha, float min_weight, float max_weight);

  std::vector<Source> sources_;
  std::vector<MixerSpec> mixer_specs_;
  std::vector<MixerInput> layer_inputs_; // sized by num_layers
  std::vector<Mixer> mixers_layer0_;
  std::vector<Mixer> mixers_layer1_;
  Sigmoid sigmoid_;
  SSE sse_;
  LstmSpec lstm_spec_{};
  std::unique_ptr<Lstm> lstm_;
  // Source weighting state
  bool weighting_enabled_ = false;
  bool duplicate_weighted_input_ = true;
  float weight_alpha_ = 0.001f;
  float min_weight_ = 0.05f;
  float max_weight_ = 1.5f;
  std::vector<float> source_weights_;      // adaptive weights per source
  std::vector<float> last_source_probs_;   // last raw probability per source
};

#endif // GENERIC_PREDICTOR_H
