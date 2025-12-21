#include "generic_full_predictor.h"
#include "models/fxcmv1.h"
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include "mixer/NullMixer.hpp"

static inline unsigned int DiscretizeLocal(float p) {
  return 1 + 4094 * p;
}

GenericFullPredictor::GenericFullPredictor(const std::vector<bool> &vocab,
                                           int ppmd_order, int ppmd_mem_mb)
    : manager_(), sigmoid_(100001), vocab_(vocab), ppmd_order_(ppmd_order),
      ppmd_mem_mb_(ppmd_mem_mb) {
  stats_file_.open("predictor_stats.csv", std::ios::out | std::ios::trunc);
  if (stats_file_.is_open()) {
    WriteStatsHeader();
  }
}

void GenericFullPredictor::InitFxcm() {
  fxcm_model_ = std::make_unique<FXCM>();
}

unsigned long long GenericFullPredictor::GetNumModels() {
  unsigned long long num = 0;

  num += bracket_model_->NumOutputs();
  num += fxcm_model_->NumOutputs();
  num += direct_models_.size();
  num += match_models_.size();
  num += indirect_ns_models_.size();
  num += indirect_r_models_.size();
  num += byte_model_->NumOutputs();
  num += byte_mixer_->NumOutputs();
  return num;
}

void GenericFullPredictor::AddMixer(int                       layer,
                                    const unsigned long long &context,
                                    float                     learning_rate) {
  if (layer == 0) {
    mixer_0_.emplace_back(layers_[layer].Inputs(), layers_[layer].ExtraInputs(),
                          context, learning_rate, mixer_0_.size());
  } else {
    mixer_1_.emplace_back(layers_[layer].Inputs(), layers_[layer].ExtraInputs(),
                          context, learning_rate, mixer_1_.size());
  }
}

void GenericFullPredictor::AddBracket() {
  bracket_model_.emplace(manager_.bit_context_, 200, 10, 100000, vocab_);
  const Context &context =
      manager_.AddBracketContext(manager_.bit_context_, 256, 15);
  direct_models_.emplace_back(context.GetContext(), manager_.bit_context_, 30,
                              0, context.Size());
  indirect_ns_models_.emplace_back(manager_.nonstationary_,
                                   context.GetContext(), manager_.bit_context_,
                                   300, manager_.shared_map_);
}

void GenericFullPredictor::AddPPMD() {
  byte_model_.emplace(ppmd_order_, ppmd_mem_mb_, manager_.bit_context_, vocab_);
}

void GenericFullPredictor::AddWord() {
  float                                  delta        = 200;
  std::vector<std::vector<unsigned int>> model_params = {
      {0},    {0, 1}, {1},       {1, 2},    {1, 3},
      {2, 3}, {3, 4}, {1, 2, 4}, {2, 3, 4}, {2}};
  for (const auto &params : model_params) {
    const Context &context = manager_.AddSparseContext(manager_.words_, params);
    indirect_ns_models_.emplace_back(
        manager_.nonstationary_, context.GetContext(), manager_.bit_context_,
        delta, manager_.shared_map_);
  }

  std::vector<std::vector<unsigned int>> model_params2 = {
      {0}, {1}, {1, 3}, {1, 2, 3}, {7, 2}};
  for (const auto &params : model_params2) {
    const Context &context = manager_.AddSparseContext(manager_.words_, params);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
                               manager_.bit_context_, 200, 0.5, 2000000,
                               &(manager_.longest_match_));
    if (params[0] == 1 && params.size() == 1) {
      indirect_r_models_.emplace_back(manager_.run_map_, context.GetContext(),
                                      manager_.bit_context_, delta,
                                      manager_.shared_map_);
    }
  }
}

void GenericFullPredictor::AddMatch() {
  float                         delta        = 0.5;
  int                           limit        = 200;
  unsigned long long            max_size     = 2000000;
  std::vector<std::vector<int>> model_params = {
      {0, 8}, {1, 8}, {7, 4}, {11, 3}, {13, 2},
  };

  for (const auto &params : model_params) {
    const Context &context = manager_.AddContextHashContext(
        manager_.bit_context_, params[0], params[1]);
    match_models_.emplace_back(
        manager_.history_, context.GetContext(), manager_.bit_context_, limit,
        delta, std::min(max_size, context.Size()), &(manager_.longest_match_));
  }
}

void GenericFullPredictor::AddDoubleIndirect() {
  float delta = 400;
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind1,
                                   manager_.bit_context_, delta,
                                   manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind2,
                                   manager_.bit_context_, delta,
                                   manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind3,
                                   manager_.bit_context_, delta,
                                   manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind5,
                                   manager_.bit_context_, delta,
                                   manager_.shared_map_);
}

void GenericFullPredictor::AddMixers() {
  unsigned int vocab_size = 0;
  for (unsigned int i = 0; i < vocab_.size(); ++i) {
    if (vocab_[i])
      ++vocab_size;
  }
  byte_mixer_.emplace(1, manager_.bit_context_, vocab_, vocab_size,
                      new Lstm(vocab_size, vocab_size, 200, 1, 128, 0.03, 10));

  // Initialize predictor names for statistics
  predictor_names_.clear();
  predictor_names_.push_back("bracket_model");
  predictor_names_.push_back("fxcm_model");
  for (size_t i = 0; i < direct_models_.size(); ++i) {
    predictor_names_.push_back("direct_" + std::to_string(i));
  }
  for (size_t i = 0; i < match_models_.size(); ++i) {
    predictor_names_.push_back("match_" + std::to_string(i));
  }
  for (size_t i = 0; i < indirect_ns_models_.size(); ++i) {
    predictor_names_.push_back("indirect_ns_" + std::to_string(i));
  }
  for (size_t i = 0; i < indirect_r_models_.size(); ++i) {
    predictor_names_.push_back("indirect_r_" + std::to_string(i));
  }
  predictor_names_.push_back("byte_model");
  predictor_names_.push_back("byte_mixer");
  predictor_names_.push_back("final_mix");
  
  // Initialize stats structure
  predictor_stats_.resize(predictor_names_.size());
  for (size_t i = 0; i < predictor_names_.size(); ++i) {
    predictor_stats_[i].name = predictor_names_[i];
  }
  
  // Now rewrite the header with actual predictor names
  if (stats_file_.is_open()) {
    stats_file_.seekp(0);
    WriteStatsHeader();
  }

  for (int i = 0; i < 2; ++i) {
    layers_.emplace_back(sigmoid_, 1.0e-4);
  }

  unsigned long long input_size = GetNumModels();
  std::cout << "num models " << input_size << "\n";
  layers_[0].SetNumModels(input_size);

  AddMixer(0, manager_.mx9, 0.005);
  AddMixer(0, manager_.mx10, 0.0005);
  AddMixer(0, manager_.mx11, 0.005);
  AddMixer(0, manager_.mx12, 0.0005);
  AddMixer(0, manager_.mx13, 0.005);
  AddMixer(0, manager_.mxx, 0.001);
  AddMixer(0, manager_.recent_bytes_[2], 0.002);
  AddMixer(0, manager_.line_break_, 0.0007);
  AddMixer(0, manager_.longest_match_, 0.0005);
  AddMixer(0, manager_.mx19cxt, 0.002);
  AddMixer(0, manager_.auxiliary_context_, 0.0005);
  AddMixer(0, manager_.mx18, 0.001);
  AddMixer(0, manager_.mx7, 0.001);
  AddMixer(0, manager_.wordscxt, 0.005);
  AddMixer(0, manager_.b2streamcxt, 0.001);
  AddMixer(0, manager_.mx5, 0.001);
  AddMixer(0, manager_.mx6, 0.005);
  AddMixer(0, manager_.b3streamcxt, 0.001);
  AddMixer(0, manager_.mx8, 0.001);
  AddMixer(0, manager_.mx17, 0.005);
  AddMixer(0, manager_.mx16, 0.005);
  AddMixer(0, manager_.mx14, 0.005);
  AddMixer(0, manager_.mx15, 0.005);

  input_size = mixer_0_.size() + auxiliary_size_;
  layers_[1].SetNumModels(input_size);

  AddMixer(1, manager_.zero_context_, 0.0003);

  layers_[0].SetExtraInputSize(mixer_0_.size());

  if (mw_enabled_) {
    mw_weights_.assign(mixer_0_.size(), 1.0f);
    mw_last_probs_.assign(mixer_0_.size(), 0.5f);
  }
}

extern int   lstmpr, lstmex;
static float s_byte_mixer_output = 0.0f;

float        GenericFullPredictor::Predict() {
  unsigned int input_index          = 0;
  unsigned int stats_index          = 0;
  auto         bracket_model_output = bracket_model_->Predict()[0];
  layers_[0].SetInput(input_index++, bracket_model_output);
  if (stats_index < predictor_stats_.size()) {
    predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(bracket_model_output);
  }

  const auto &fxcm_model_outputs = fxcm_model_->Predict();
  for (unsigned int j = 0; j < fxcm_model_outputs.size(); ++j) {
    layers_[0].SetInput(input_index, fxcm_model_outputs[j]);
    ++input_index;
  }
  auto fxcm_model_index = input_index - 1;
  if (stats_index < predictor_stats_.size()) {
    predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(fxcm_model_outputs[0]);
  }

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    const Eigen::VectorXf &outputs = direct_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
    if (stats_index < predictor_stats_.size()) {
      predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(outputs[0]);
    }
  }

  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    const Eigen::VectorXf &outputs = match_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
    if (stats_index < predictor_stats_.size()) {
      predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(outputs[0]);
    }
  }

  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    const Eigen::VectorXf &outputs = indirect_ns_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
    if (stats_index < predictor_stats_.size()) {
      predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(outputs[0]);
    }
  }

  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    const Eigen::VectorXf &outputs = indirect_r_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
    if (stats_index < predictor_stats_.size()) {
      predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(outputs[0]);
    }
  }
  auto byte_model_pred = byte_model_->Predict()[0];
  layers_[0].SetInput(input_index++, byte_model_pred);
  if (stats_index < predictor_stats_.size()) {
    predictor_stats_[stats_index++].last_prediction = Sigmoid::Logistic(byte_model_pred);
  }

  float byte_mixer_override = -1;

  if (s_byte_mixer_output == 0 || s_byte_mixer_output == 1)
    byte_mixer_override = s_byte_mixer_output;
  layers_[0].SetInput(input_index++, s_byte_mixer_output);
  auto  byte_mixer_index = input_index - 1;
  if (stats_index < predictor_stats_.size()) {
    predictor_stats_[stats_index++].last_prediction = s_byte_mixer_output;
  }

  float auxiliary_average =
      Sigmoid::Logistic(layers_[0].Inputs()[fxcm_model_index]) +
      Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
  auxiliary_average /= auxiliary_size_;
  manager_.auxiliary_context_ = auxiliary_average * 15;

  for (unsigned int i = 0; i < mixer_0_.size(); ++i) {
    float logit = mixer_0_[i].Mix();
    if (mw_enabled_) {
      if (mw_weights_.size() != mixer_0_.size()) { // late initialization safety
        mw_weights_.assign(mixer_0_.size(), 1.0f);
        mw_last_probs_.assign(mixer_0_.size(), 0.5f);
      }
      float prob        = Sigmoid::Logistic(logit);
      mw_last_probs_[i] = prob;
    }
    float scaled_logit = mw_enabled_ ? (logit * mw_weights_[i]) : logit;
    layers_[0].SetExtraInput(i, scaled_logit);
    layers_[1].SetStretchedInput(i, scaled_logit);
  }
  layers_[1].SetStretchedInput(mixer_0_.size(),
                                      layers_[0].Inputs()[fxcm_model_index]);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1,
                                      layers_[0].Inputs()[byte_mixer_index]);

  float p = Sigmoid::Logistic(mixer_1_[0].Mix());
  p       = sse_.Predict(p);
  
  // Track final mixed prediction
  if (stats_index < predictor_stats_.size()) {
    predictor_stats_[stats_index].last_prediction = p;
  }
  
  if (byte_mixer_override >= 0) {
    return byte_mixer_override;
  }
  return p;
}

void GenericFullPredictor::Perceive(int bit) {
  // Update statistics for all predictors
  for (size_t i = 0; i < predictor_stats_.size(); ++i) {
    float pred = predictor_stats_[i].last_prediction;
    // Consider a "hit" if prediction was closer to the actual bit than 0.5
    bool correct = (bit == 1 && pred > 0.5) || (bit == 0 && pred < 0.5);
    if (correct) {
      predictor_stats_[i].hits++;
    } else {
      predictor_stats_[i].misses++;
    }
  }
  
  total_bits_++;
  if (total_bits_ % stats_interval_ == 0) {
    WriteStatsSnapshot();
  }

  bracket_model_->Perceive(bit);

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }

  byte_model_->Perceive(bit);

  byte_mixer_->Perceive(bit);

  for (auto &mixer : mixer_0_) {
    mixer.Perceive(bit);
  }
  for (auto &mixer : mixer_1_) {
    mixer.Perceive(bit);
  }

  sse_.Perceive(bit);

  if (mw_enabled_) {
    for (size_t i = 0; i < mw_weights_.size(); ++i) {
      float err     = std::fabs((float)bit - mw_last_probs_[i]);
      float quality = 1.0f - err;
      mw_weights_[i] =
          mw_weights_[i] * (1.0f - mw_alpha_) + mw_alpha_ * quality;
      if (mw_weights_[i] < mw_min_)
        mw_weights_[i] = mw_min_;
      if (mw_weights_[i] > mw_max_)
        mw_weights_[i] = mw_max_;
    }
  }

  bool byte_update = false;
  if (manager_.bit_context_ >= 128)
    byte_update = true;

  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }

    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }

    byte_model_->ByteUpdate();

    const Eigen::VectorXf &p = byte_model_->BytePredict();
    for (unsigned int j = 0; j < 256; ++j) {
      byte_mixer_->SetInput(j, p[j]);
    }

    byte_mixer_->ByteUpdate();
  }
  s_byte_mixer_output = byte_mixer_->Predict()[0];
  lstmpr              = DiscretizeLocal(s_byte_mixer_output);
  lstmex              = byte_mixer_->ex;
  fxcm_model_->Perceive(bit);
  if (byte_update)
    manager_.bit_context_ = 1;
}

void GenericFullPredictor::WriteStatsHeader() {
  if (!stats_file_.is_open()) return;
  
  stats_file_ << "bit_position";
  for (const auto& stats : predictor_stats_) {
    stats_file_ << "," << stats.name;
  }
  stats_file_ << "\n";
  stats_file_.flush();
}

void GenericFullPredictor::WriteStatsSnapshot() {
  if (!stats_file_.is_open()) return;
  
  stats_file_ << total_bits_;
  
  for (const auto& stats : predictor_stats_) {
    unsigned long long total = stats.hits + stats.misses;
    float accuracy = total > 0 ? (float)stats.hits / (float)total * 100.0f : 0.0f;
    stats_file_ << "," << accuracy;
  }
  stats_file_ << "\n";
  stats_file_.flush();
  
  // Reset counters for next interval
  for (auto& stats : predictor_stats_) {
    stats.hits = 0;
    stats.misses = 0;
  }
}

void GenericFullPredictor::Pretrain(int bit) {
  bracket_model_->Predict();
  fxcm_model_->Predict();

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Predict();
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Predict();
  }

  bracket_model_->Perceive(bit);
  fxcm_model_->Perceive(bit);

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }

  bool byte_update = false;
  if (manager_.bit_context_ >= 128)
    byte_update = true;
  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
    manager_.bit_context_ = 1;
  }
}
