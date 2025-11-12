#pragma once

#include "IPredictor.h"
#include <optional>
// TModel extends Model
template <typename TModel> class SimplePredictor : public IPredictor {
private:
  std::optional<TModel> _model;
  int                   _outputIndex;

public:
  SimplePredictor() : _model(std::nullopt), _outputIndex(0) {}
  // ctor with args to emplace model
  template <typename... TArgs>
  SimplePredictor(int outputIndex, TArgs &&...args)
      : _model(std::in_place, std::forward<TArgs>(args)...),
        _outputIndex(outputIndex) {}
  float Predict() override {
    if (_model.has_value()) {
      auto result = _model->Predict()[_outputIndex];
      return result;
    } else {
      return 0.5f;
    }
  }
  void Perceive(int bit) override {
    if (!_model.has_value()) {
      return;
    }
    _model->Perceive(bit);
  }
  void Pretrain(int bit) override {}
};

template <typename TModel> class SimpleByteModelPredictor : public IPredictor {
private:
  std::optional<TModel> _model;
  int                   _outputIndex;
  size_t                _byteUpdateEveryBits;
  size_t                _bits;

public:
  // ctor with args to emplace model
  template <typename... TArgs>
  SimpleByteModelPredictor(int outputIndex, size_t byteUpdateEveryBits, TArgs &&...args)
      : _model(std::in_place, std::forward<TArgs>(args)...),
        _outputIndex(outputIndex), _byteUpdateEveryBits(byteUpdateEveryBits), _bits(0) {}
  float Predict() override {
    if (_model.has_value()) {
      auto result = _model->Predict()[_outputIndex];
      return result;
    } else {
      return 0.5f;
    }
  }
  void Perceive(int bit) override {
    if (!_model.has_value()) {
      return;
    }
    _model->Perceive(bit);
    _bits++;
    if (_bits % _byteUpdateEveryBits == 0) {
      _model->ByteUpdate();
    }
  }
  void Pretrain(int bit) override {}
};