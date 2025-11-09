#include "random.hpp"
#include <random>

std::default_random_engine            generator;
std::uniform_real_distribution<float> distribution(0.0, 1.0);

/*
    Sets the seed for the random number generator.
*/
void set_seed(unsigned int seed) {
  generator.seed(seed);
}

/*
    Returns a random float in [0.0, 1.0).
*/
float get_uniform_random() {
  return distribution(generator);
}

/* Returns random like from stdlib */
int get_random() {
  return generator();
}