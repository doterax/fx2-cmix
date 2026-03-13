#ifndef GENERIC_FULL_PREDICTOR_H
#define GENERIC_FULL_PREDICTOR_H

#include "IPredictor.h"

#include "context-manager.h"
#include "contexts/bit-context.h"
#include "contexts/bracket-context.h"
#include "contexts/context-hash.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval-hash.h"
#include "contexts/interval.h"
#include "contexts/sparse.h"
#include "mixer/byte-mixer.h"
#include "mixer/lstm.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/sigmoid.h"
#include "mixer/sse.h"
#include "models/bracket.h"
#include "models/byte-model.h"
#include "models/direct-hash.h"
#include "models/direct.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/model.h"
#include "models/ppmd.h"
#include "models/url-model.h"

#include "ds/SmallVector.h"

#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

// A self-contained replica of Predictor that does not depend on the Predictor
// class. Implements the same model wiring and runtime behavior so results match
// FULL mode.
class FXCM; // forward declaration in global scope

class GenericFullPredictor : public IPredictor {
public:
  GenericFullPredictor(const std::vector<bool> &vocab, int ppmd_order = 25,
                       int ppmd_mem_mb = 1024, int lstm_bptt_depth = 0);
  // Prediction API
  float Predict() override;
  void  Perceive(int bit) override;
  void  Pretrain(int bit) override;
  ~GenericFullPredictor() override;

  // Explicit initialization API (moved out of ctor to allow flexible wiring)
  void InitFxcm();
  void AddPPMD();
  void AddURLModel();
  void AddBracket();
  void AddWord();
  void AddMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void SetAuxiliarySize(size_t sz) { auxiliary_size_ = sz; }
  // Experimental: adaptively weight layer-0 mixer outputs
  void EnableMixerWeighting(float alpha = 0.001f, float min_w = 0.05f,
                            float max_w = 1.5f) {
    mw_enabled_ = true;
    mw_alpha_   = alpha;
    mw_min_     = min_w;
    mw_max_     = max_w;
    if (mixer_0_.size() > 0) { // initialize immediately if mixers already built
      mw_weights_.assign(mixer_0_.size(), 1.0f);
      mw_last_probs_.assign(mixer_0_.size(), 0.5f);
    }
  }

private:
  unsigned long long GetNumModels();
  void               AddMixer(int layer, const unsigned long long &context,
                              float learning_rate);

  llvm::SmallVector<Indirect<Nonstationary>, 30 - 7>
                                         indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_;  // run map
  llvm::SmallVector<Direct, 1>           direct_models_;
  llvm::SmallVector<Match, 10>           match_models_;

  std::optional<Bracket>                 bracket_model_;
  size_t auxiliary_size_ = 2; // 0 -> fxcm/url_model, 1 -> byte_mixer
  SSE    sse_;
  llvm::SmallVector<MixerInput, 2> layers_;
  llvm::SmallVector<Mixer, 23>     mixer_0_;
  llvm::SmallVector<Mixer, 1>      mixer_1_;
  std::vector<unsigned int>        auxiliary_;
  ContextManager                   manager_;
  Sigmoid                          sigmoid_;
  std::optional<PPMD::PPMD>        byte_model_;
  std::optional<ByteMixer>         byte_mixer_;
  std::optional<URLModel>          url_model_;
  std::vector<bool>                vocab_;
  std::unique_ptr<FXCM>            fxcm_model_;

  // Config
  int ppmd_order_      = 25;
  int ppmd_mem_mb_     = 1024;
  int lstm_bptt_depth_ = 0;

  // ByteMixer output override
  float byte_mixer_output_ = 0.0f;

  // Mixer weighting state (experimental)
  bool               mw_enabled_ = false;
  float              mw_alpha_   = 0.001f;
  float              mw_min_     = 0.05f;
  float              mw_max_     = 1.5f;
  std::vector<float> mw_weights_;
  std::vector<float> mw_last_probs_;

  // Predictor statistics tracking
  struct PredictorStats {
    std::string        name;
    unsigned long long hits            = 0;
    unsigned long long misses          = 0;
    float              last_prediction = 0.5f;
  };
  std::vector<PredictorStats> predictor_stats_;
  std::vector<std::string>    predictor_names_;
  unsigned long long          total_bits_ = 0;
  unsigned long long stats_interval_      = 8000; // Write stats every N bits
  std::ofstream      stats_file_;
  void               WriteStatsHeader();
  void               WriteStatsSnapshot();
};

#endif
