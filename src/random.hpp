#pragma once
/*
    Sets the seed for the random number generator.
*/
void set_seed(unsigned int seed);
/*
    Returns a random float in [0.0, 1.0).
*/
float get_uniform_random();

/* Returns random like from stdlib */
int get_random();