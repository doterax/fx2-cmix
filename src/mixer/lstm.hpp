#include "lstm.h"

#include <numeric>
#include <fstream>
#include <iostream>

inline Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    hidden_(Eigen::VectorXf::Zero(num_cells * num_layers + 1)), 
    hidden_error_(Eigen::VectorXf::Zero(num_cells)),
    layer_input_(horizon, std::vector<Eigen::VectorXf>(num_layers)),
    output_layer_(horizon, std::vector<Eigen::VectorXf>(output_size)),
    output_(horizon, Eigen::VectorXf::Constant(output_size, 1.0 / output_size)),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), input_size_(input_size), output_size_(output_size) {
  hidden_[hidden_.size() - 1] = 1;
  
  for (int epoch = 0; epoch < horizon; ++epoch) {
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i] = Eigen::VectorXf::Zero(input_size + 1 + num_cells * 2);
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
    
    for (unsigned int i = 0; i < output_size; ++i) {
      output_layer_[epoch][i] = Eigen::VectorXf::Zero(num_cells * num_layers + 1);
    }
  }
  
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, gradient_clip, learning_rate);
  }
}

inline Lstm::~Lstm() {
  //SaveToDisk("lstm.dat");
}
/*
inline void Lstm::SaveToDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ofstream os(path, std::ios::binary | std::ios::out);
  if (!os.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    os.write(reinterpret_cast<const char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        os.write(reinterpret_cast<const char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  os.close();
}

inline void Lstm::LoadFromDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ifstream is(path, std::ios::binary | std::ios::in);
  if (!is.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    is.read(reinterpret_cast<char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        is.read(reinterpret_cast<char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  is.close();
}
*/
inline void Lstm::SetInput(const Eigen::VectorXf& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    layer_input_[epoch_][i].head(input_size_) = input.head(input_size_);
  }
}

inline Eigen::VectorXf& Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        for (unsigned int i = 0; i < output_size_; ++i) {
//          float error = 0;
//          if (i == input_history_[epoch]) error = output_[epoch][i] - 1;
//          else error = output_[epoch][i];
          float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
          for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] += output_layer_[epoch][i][j + offset] * error;
          }
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_);
      }
    }
  }

  for (unsigned int i = 0; i < output_size_; ++i) {
//    float error = 0;
//    if (i == input) error = output_[last_epoch][i] - 1;
//    else error = output_[last_epoch][i];
    float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
    output_layer_[epoch_][i] = output_layer_[last_epoch][i];
    output_layer_[epoch_][i] -= learning_rate_ * error * hidden_;
  }
  return Predict(input);
}

inline Eigen::VectorXf& Lstm::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    layer_input_[epoch_][i].segment(input_size_, num_cells_) = 
        hidden_.segment(i * num_cells_, num_cells_);
    layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
        num_cells_);
    if (i < layers_.size() - 1) {
      layer_input_[epoch_][i + 1].segment(num_cells_ + input_size_, num_cells_) = 
          hidden_.segment(i * num_cells_, num_cells_);
    }
  }
  for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = hidden_.dot(output_layer_[epoch_][i]);
    output_[epoch_][i] = exp(sum);
  }
  float total = output_[epoch_].sum();
  output_[epoch_] /= total;
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = input;
  return output_[epoch];
}

