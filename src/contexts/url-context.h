#ifndef URL_CONTEXT_H
#define URL_CONTEXT_H

#include "context.h"

class URLContext : public Context {
 public:
  URLContext(const unsigned int& bit_context, const unsigned int& url_state,
             const unsigned int& url_position, int max_url_length);
  void Update();
  bool IsEqual(Context* c) const;

 private:
  const unsigned int& byte_;
  const unsigned int& url_state_;
  const unsigned int& url_position_;
  int max_url_length_;
};

#endif
