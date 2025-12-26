#ifndef URL_MODEL_H
#define URL_MODEL_H

#include "byte-model.h"
#include "ppmd.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

// URL Model: Specialized predictor for URL sequences
//
// Detection Strategy:
// - Tracks byte sequences for URL patterns (http://, https://, www., ftp://)
// - Analyzes context before URLs (common patterns: href=", src=", <a, etc.)
// - Maintains state machine: SCANNING -> DETECTED -> IN_URL -> AFTER_URL
//
// Prediction Strategy:
// - Before URL: uses context patterns and markov chain
// - Inside URL: uses dedicated small PPMD (order 6, 16MB) for URL-specific
// stats
// - After URL: returns to baseline
//
// Memory: ~{ppmdMemoryMb} MB for PPMD + small buffers for context tracking
class URLModel : public ByteModel {
public:
  URLModel(int ppmdOrder, int ppmdMemoryMb, bool ppmdVerbose,
           const unsigned int &bit_context, const std::vector<bool> &vocab,
           unsigned int &url_state, unsigned int &url_position);
  ~URLModel();

  void ByteUpdate() override;

  // Statistics for analysis
  struct Stats {
    unsigned long long total_bytes;
    unsigned long long url_bytes;
    unsigned long long urls_detected;
    unsigned long long false_positives;
  };

  const Stats &GetStats() const { return stats_; }

  bool IsInURL() const { return state_ == IN_URL; }
  bool ShouldContribute() const { return state_ == IN_URL || state_ == DETECTED; }

private:
  enum State {
    SCANNING, // Normal text, looking for URL indicators
    DETECTED, // URL pattern detected, preparing PPMD
    IN_URL,   // Inside URL, using PPMD
    AFTER_URL // Just exited URL, cooldown period
  };

  // URL pattern detection
  bool CheckURLPattern();
  bool IsURLChar(unsigned char c) const;
  bool IsURLStart() const;

  // Context analysis
  void  UpdateContextBuffer(unsigned char c);
  float GetContextBoost() const;

  // State management
  void                EnterURL();
  void                ExitURL();
  void                UpdateState();

  const unsigned int &byte_;
  unsigned int &url_state_;    // Reference to manager's url_state
  unsigned int &url_position_; // Reference to manager's url_position
  State               state_;

  // Pattern matching buffer (last 16 bytes for context)
  static const int CONTEXT_SIZE = 16;
  unsigned char    context_buffer_[CONTEXT_SIZE];
  int              context_pos_;

  // URL detection buffer (last 8 bytes for pattern matching)
  static const int PATTERN_SIZE = 8;
  unsigned char    pattern_buffer_[PATTERN_SIZE];
  int              pattern_pos_;

  // Small PPMD for URL content (order 6, 16MB)
  std::unique_ptr<PPMD::PPMD> url_ppmd_;

  // Counters
  int         bytes_in_url_;
  int         bytes_since_url_;
  Stats       stats_;
  std::string current_url_;

  // Base probabilities for different states
  Eigen::VectorXf base_probs_;
  Eigen::VectorXf url_probs_;

  // Common URL prefixes for detection
  static const char *const URL_PATTERNS[];
  static const int         NUM_URL_PATTERNS;

  // Common pre-URL context patterns
  static const char *const CONTEXT_PATTERNS[];
  static const int         NUM_CONTEXT_PATTERNS;
};

#endif
