#include "url-context.h"

URLContext::URLContext(const unsigned int& bit_context,
                       const unsigned int& url_state,
                       const unsigned int& url_position,
                       int max_url_length)
    : byte_(bit_context), url_state_(url_state),
      url_position_(url_position), max_url_length_(max_url_length) {
  context_ = 0;
  // Size: state (4 states) * position in URL (max_url_length)
  size_ = 4 * max_url_length_;
}

void URLContext::Update() {
  // Read URL state from ContextManager
  // State encoding: 0=SCANNING, 1=DETECTED, 2=IN_URL, 3=AFTER_URL
  unsigned int state = url_state_;
  unsigned int pos = url_position_;
  
  // Clamp position to max
  if (pos >= max_url_length_) {
    pos = max_url_length_ - 1;
  }
  
  // Context = state * max_url_length + position
  context_ = state * max_url_length_ + pos;
}

bool URLContext::IsEqual(Context* c) const {
  URLContext* p = dynamic_cast<URLContext*>(c);
  if (!p) return false;
  if (size_ == p->size_ && max_url_length_ == p->max_url_length_) {
    return true;
  }
  return false;
}
