#pragma once

class IPredictor {
public:
  virtual float Predict()         = 0;
  virtual void  Perceive(int bit) = 0;
  virtual void  Pretrain(int bit) = 0;
};
