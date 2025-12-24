#include "url-model.h"
#include <algorithm>
#include <cstring>
#include <iostream>

typedef unsigned char byte;

// URL pattern signatures for detection
const char *const URLModel::URL_PATTERNS[] = {
    "http://", "https://", "ftp://", "www.",
    "://", // Generic protocol
};
const int URLModel::NUM_URL_PATTERNS = 5;

// Common context patterns before URLs (HTML, markdown, etc.)
const char *const URLModel::CONTEXT_PATTERNS[] = {
    "href=\"", "src=\"", "<a ", "<img ", "[url]", "](http", " http", "\nhttp",
};
const int URLModel::NUM_CONTEXT_PATTERNS = 8;

URLModel::URLModel(int ppmdOrder, int ppmdMemoryMb, bool ppmdVerbose,
                   const unsigned int      &bit_context,
                   const std::vector<bool> &vocab)
    : ByteModel(vocab), byte_(bit_context), state_(SCANNING), context_pos_(0),
      pattern_pos_(0), bytes_in_url_(0), bytes_since_url_(0),
      base_probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)),
      url_probs_(Eigen::VectorXf::Constant(256, 1.0f / 256)) {

  // Initialize buffers
  memset(context_buffer_, 0, CONTEXT_SIZE);
  memset(pattern_buffer_, 0, PATTERN_SIZE);

  // Initialize statistics
  stats_.total_bytes     = 0;
  stats_.url_bytes       = 0;
  stats_.urls_detected   = 0;
  stats_.false_positives = 0;

  // Create small PPMD for URL prediction
  // URLs have specific character distributions that PPMD can learn
  url_ppmd_.reset(
      new PPMD::PPMD(ppmdOrder, ppmdMemoryMb, bit_context, vocab, ppmdVerbose));

  // Initialize probability vectors
  probs_ = base_probs_;
}

URLModel::~URLModel() {}

bool URLModel::IsURLChar(unsigned char c) const {
  // Valid URL characters: alphanumeric, dash, dot, slash, colon,
  // underscore, question mark, ampersand, equals, percent, hash
  unsigned char ch = static_cast<unsigned char>(c);
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= '0' && ch <= '9') || ch == '-' || ch == '.' || ch == '/' ||
         ch == ':' || ch == '_' || ch == '?' || ch == '&' || ch == '=' ||
         ch == '%' || ch == '#' || ch == '+' || ch == '~' || ch == '@' ||
         ch == '!' || ch == '*' || ch == '(' || ch == ')' || ch == '[' ||
         ch == ']' || ch == ';';
}

bool URLModel::IsURLStart() const {
  // Check if pattern buffer contains URL start signature
  for (int i = 0; i < NUM_URL_PATTERNS; i++) {
    const char *pattern = URL_PATTERNS[i];
    int         len     = strlen(pattern);

    if (len > PATTERN_SIZE)
      continue;

    // Check if pattern matches end of buffer
    bool match = true;
    for (int j = 0; j < len; j++) {
      int buf_idx = (pattern_pos_ - len + j + PATTERN_SIZE) % PATTERN_SIZE;
      if (static_cast<unsigned char>(pattern_buffer_[buf_idx]) !=
          static_cast<unsigned char>(pattern[j])) {
        match = false;
        break;
      }
    }

    if (match)
      return true;
  }

  return false;
}

bool URLModel::CheckURLPattern() {
  return IsURLStart();
}

void URLModel::UpdateContextBuffer(unsigned char c) {
  context_buffer_[context_pos_] = c;
  context_pos_                  = (context_pos_ + 1) % CONTEXT_SIZE;

  pattern_buffer_[pattern_pos_] = c;
  pattern_pos_                  = (pattern_pos_ + 1) % PATTERN_SIZE;
}

float URLModel::GetContextBoost() const {
  // Check if context buffer contains pre-URL patterns
  for (int i = 0; i < NUM_CONTEXT_PATTERNS; i++) {
    const char *pattern = CONTEXT_PATTERNS[i];
    int         len     = strlen(pattern); // TODO: optimize with precomputed lengths

    if (len > CONTEXT_SIZE)
      continue;

    // Check if pattern matches end of context buffer
    bool match = true;
    for (int j = 0; j < len; j++) {
      int buf_idx = (context_pos_ - len + j + CONTEXT_SIZE) % CONTEXT_SIZE;
      if (static_cast<unsigned char>(context_buffer_[buf_idx]) !=
          static_cast<unsigned char>(pattern[j])) {
        match = false;
        break;
      }
    }

    if (match)
      return 2.0f; // Strong boost when context matches
  }

  return 1.0f;
}

void URLModel::EnterURL() {
  state_        = IN_URL;
  bytes_in_url_ = 0;
  stats_.urls_detected++;
  current_url_.clear();
}

void URLModel::ExitURL() {
  if (bytes_in_url_ < 8) {
    // Too short, probably false positive
    stats_.false_positives++;
  } else {
    stats_.url_bytes += bytes_in_url_;
  }

  state_           = AFTER_URL;
  bytes_since_url_ = 0;
}

void URLModel::UpdateState() {
  byte current = byte_;

  switch (state_) {
  case SCANNING:
    if (CheckURLPattern()) {
      state_ = DETECTED;
    }
    break;

  case DETECTED:
    // Confirmed URL start, enter URL mode
    EnterURL();
    current_url_ += static_cast<char>(current);
    break;

  case IN_URL:
    bytes_in_url_++;

    // Check for URL end conditions
    if (!IsURLChar(current)) {
      // Whitespace, quote, bracket, etc. - URL ended
      if (current == ' ' || current == '\n' || current == '\r' ||
          current == '"' || current == '\'' || current == '<' ||
          current == '>' || current == ')' || current == ']' ||
          current == '}' || current == ',' || current == ';') {
        ExitURL();
      }
    } else {
      // Still in URL
      current_url_ += static_cast<char>(current);
    }
    break;

  case AFTER_URL:
    bytes_since_url_++;
    if (bytes_since_url_ > 10) {
      // Cooldown period over, return to scanning
      state_ = SCANNING;
    } else if (CheckURLPattern()) {
      // Another URL detected quickly
      state_ = DETECTED;
    }
    break;
  }
}

void URLModel::ByteUpdate() {
  stats_.total_bytes++;

  // Update context tracking
  UpdateContextBuffer(byte_);

  // Update state machine
  UpdateState();

  // Generate predictions based on state
  if (state_ == IN_URL) {
    // Inside URL: use PPMD for URL-specific predictions
    url_ppmd_->ByteUpdate();
    url_probs_ = url_ppmd_->BytePredict();

    // URL characters should have higher probability
    for (int i = 0; i < 256; i++) {
      if (IsURLChar(i)) {
        probs_[i] = url_probs_[i] * 1.5f; // Boost URL chars
      } else {
        probs_[i] = url_probs_[i] * 0.1f; // Suppress non-URL chars
      }
    }

    // Normalize
    float sum = probs_.sum();
    if (sum > 0)
      probs_ /= sum;

  } else if (state_ == DETECTED) {
    // URL just detected: boost URL start characters
    probs_ = base_probs_;

    // Boost common URL start chars: letters, numbers
    for (int i = 0; i < 256; i++) {
      if ((i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z') ||
          (i >= '0' && i <= '9') || i == '/') {
        probs_[i] *= 3.0f;
      }
    }

    // Apply context boost
    float boost = GetContextBoost();
    if (boost > 1.0f) {
      probs_ *= boost;
    }

    // Normalize
    float sum = probs_.sum();
    if (sum > 0)
      probs_ /= sum;

  } else {
    // Normal scanning: use baseline with slight boost for URL patterns
    probs_ = base_probs_;

    // If we see patterns like "http", "www", boost those chars
    if (CheckURLPattern()) {
      for (int i = 0; i < 256; i++) {
        if (i == ':' || i == '/' || i == '.') {
          probs_[i] *= 2.0f;
        }
      }
    }

    // Normalize
    float sum = probs_.sum();
    if (sum > 0)
      probs_ /= sum;
  }

  // Ensure minimum probabilities for all valid chars
  for (int i = 0; i < 256; i++) {
    if (vocab_[i] && probs_[i] < 1e-7f) {
      probs_[i] = 1e-7f;
    } else if (!vocab_[i]) {
      probs_[i] = 0;
    }
  }

  ByteModel::ByteUpdate();
}
