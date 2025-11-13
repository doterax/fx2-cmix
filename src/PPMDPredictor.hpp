#ifndef PPMD_PREDICTOR_HPP
#define PPMD_PREDICTOR_HPP

#include "models/ppmd.h"
#include <iostream>
#include <memory>
#include <vector>
#include "IPredictor.h"
/**
 * PPMDPredictor - Isolated PPMD predictor for performance analysis and tuning
 *
 * This class provides a minimal predictor that uses only PPMD model,
 * allowing for isolated performance testing, profiling, and parameter tuning.
 *
 * Usage:
 *   PPMDPredictor predictor(order, memory_mb);
 *   float p = predictor.Predict();
 *   predictor.Perceive(bit);
 */
class PPMDPredictor: public IPredictor {
public:
  /**
   * Constructor
   *
   * @param order PPMD model order (typical: 20-30, higher = better compression
   * but slower)
   * @param memory_mb Memory allocation in megabytes (value << 20 bytes
   * allocated) Examples: 256 = 256 MB, 512 = 512 MB, 1024 = 1 GB, 2048 = 2 GB
   * @param vocab Optional vocabulary filter (default: all 256 bytes enabled)
   */
  PPMDPredictor(int order = 25, int memory_mb = 1024,
                const std::vector<bool> &vocab = std::vector<bool>(256, true))
      :  bit_context_(1), vocab_(vocab), order_(order), memory_mb_(memory_mb) {

    // Initialize PPMD model - it holds a reference to bit_context_
    ppmd_model_ =
        std::make_unique<PPMD::PPMD>(order_, memory_mb_, bit_context_, vocab_);

    std::cout << "PPMDPredictor initialized:" << std::endl;
    std::cout << "  PPMD order: " << order_ << std::endl;
    std::cout << "  Memory: " << memory_mb_ << " MB (" << (memory_mb_ / 1024.0)
              << " GB)" << std::endl;
    std::cout << "  Vocab size: " << CountVocab() << " bytes" << std::endl;
    std::cout << "  Estimated memory usage: ~" << memory_mb_ << " MB"
              << std::endl;
  }

  /**
   * Predict the probability of the next bit being 1
   *
   * @return Probability in range [0.0, 1.0]
   */
  float Predict() {
    // Get PPMD prediction
    const Eigen::VectorXf &predictions = ppmd_model_->Predict();
    return predictions[0];
  }

  /**
   * Update the model with the actual bit value
   *
   * @param bit The actual bit value (0 or 1)
   */
  void Perceive(int bit) {
    // First, let PPMD process the bit prediction
    ppmd_model_->Perceive(bit);

    // Update bit context: shift in the new bit
    // bit_context_ goes: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256+ (complete byte)
    bit_context_ = (bit_context_ << 1) | bit;

    // Handle byte boundary - when bit_context >= 256, we have a complete byte
    if (bit_context_ >= 256) {
      // Extract the actual byte value (0-255) and reset for next byte
      unsigned int completed_byte = bit_context_ - 256;
      bit_context_ = completed_byte;  // Now bit_context_ holds the byte value
      
      // Call ByteUpdate with bit_context_ containing the completed byte value
      ppmd_model_->ByteUpdate();
      
      // Reset to start of new byte
      bit_context_ = 1;
    }
  }

  /**
   * Pretrain stub (not used in isolated PPMD testing)
   */
  void Pretrain(int bit) {
    // Not implemented for isolated testing
  }

  /**
   * Get current bit context value (for debugging)
   */
  unsigned int GetBitContext() const { return bit_context_; }

  /**
   * Get PPMD configuration
   */
  int GetOrder() const { return order_; }
  int GetMemoryMB() const { return memory_mb_; }

  /**
   * Get number of models (always 1 for PPMDPredictor)
   */
  unsigned long long GetNumModels() const { return ppmd_model_->NumOutputs(); }

  /**
   * Get statistics about prediction quality (for analysis)
   */
  struct Stats {
    unsigned long long total_bits;
    unsigned long long total_bytes;
    double             avg_probability;

    Stats() : total_bits(0), total_bytes(0), avg_probability(0.0) {}
  };

  const Stats &GetStats() const { return stats_; }

  /**
   * Reset statistics
   */
  void ResetStats() { stats_ = Stats(); }

private:
  // Bit context tracking (1-255 during byte, resets to 1 at byte boundary)
  unsigned int bit_context_;

  // Vocabulary filter
  const std::vector<bool> vocab_;

  // PPMD model configuration
  int order_;
  int memory_mb_;

  // PPMD model instance
  std::unique_ptr<PPMD::PPMD> ppmd_model_;

  // Statistics tracking
  Stats stats_;

  /**
   * Count enabled bytes in vocabulary
   */
  int CountVocab() const {
    int count = 0;
    for (bool enabled : vocab_) {
      if (enabled)
        count++;
    }
    return count;
  }
};

#endif // PPMD_PREDICTOR_HPP
