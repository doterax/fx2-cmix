#ifndef GENERIC_FULL_PREDICTOR_H
#define GENERIC_FULL_PREDICTOR_H

#include "IPredictor.h"

#include "context-manager.h"
#include "contexts/bit-context.h"
#include "contexts/bracket-context.h"
#include "contexts/combined-context.h"
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
#include "models/model.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/model.h"
#include "models/ppmd.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include <memory>
#include <optional>
#include <set>
#include <vector>

// A self-contained replica of Predictor that does not depend on the Predictor class.
// Implements the same model wiring and runtime behavior so results match FULL mode.
class FXCM; // forward declaration in global scope

class GenericFullPredictor : public IPredictor {
public:
  GenericFullPredictor(const std::vector<bool> &vocab, int ppmd_order = 25,
                       int ppmd_mem_mb = 1024);
  // Prediction API
  float Predict() override;
  void  Perceive(int bit) override;
  void  Pretrain(int bit) override;

  // Explicit initialization API (moved out of ctor to allow flexible wiring)
  void InitFxcm();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void SetAuxiliarySize(size_t sz) { auxiliary_size_ = sz; }

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
  size_t auxiliary_size_ = 2; // 0 -> fxcm, 1 -> byte_mixer
  SSE    sse_;
  llvm::SmallVector<MixerInput, 2> layers_;
  llvm::SmallVector<Mixer, 23>     mixer_0_;
  llvm::SmallVector<Mixer, 1>      mixer_1_;
  std::vector<unsigned int>        auxiliary_;
  ContextManager                   manager_;
  Sigmoid                          sigmoid_;
  std::optional<PPMD::PPMD>        byte_model_;
  std::optional<ByteMixer>         byte_mixer_;
  std::vector<bool>                vocab_;
  std::unique_ptr<FXCM>            fxcm_model_;

  // Config
  int ppmd_order_  = 25;
  int ppmd_mem_mb_ = 1024;
};

#endif
