#include "random.hpp"
#include <cstdint>

// xorshift128+ state (must be seeded with non-zero values)
static uint64_t s[2] = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

/*
    Sets the seed for the random number generator.
    Uses splitmix64 to generate good initial state from a single seed.
*/
void set_seed(unsigned int seed) {
  // Use splitmix64 to generate initial state from seed
  uint64_t z = (uint64_t(seed) ^ 0x9e3779b97f4a7c15ULL);
  
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  s[0] = z;
  
  z = (z ^ 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  s[1] = z;
}

/*
    xorshift128+ generator - fast and has good statistical properties.
    Produces consistent results across all compilers and platforms.
*/
static inline uint64_t xorshift128plus() {
  uint64_t s1 = s[0];
  const uint64_t s0 = s[1];
  s[0] = s0;
  s1 ^= s1 << 23;
  s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return s[1] + s0;
}

/*
    Returns a random float in [0.0, 1.0).
*/
float get_uniform_random() {
  // Use upper 24 bits for mantissa (float has 23-bit mantissa + 1 implicit bit)
  const uint64_t x = xorshift128plus();
  const uint32_t upper = (uint32_t)(x >> 40);
  return (upper & 0xFFFFFFU) * (1.0f / 16777216.0f); // 2^24 = 16777216
}

/* Returns random int like from stdlib */
int get_random() {
  return (int)(xorshift128plus() >> 33); // Use upper 31 bits for positive int
}