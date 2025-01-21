#include "pch.h"
#include "myrand.h"

const int gaussian_pool_capacity = 1 << 20;

MyRand<gaussian_pool_capacity> gaussian_pool;

float fast_gaussian(int random_int)
{
    return gaussian_pool.get((unsigned int)random_int);
}

xorshift32::xorshift32(uint32_t x)
{
    state = x;
}

uint32_t xorshift32::operator()()
{
    // from: https://en.wikipedia.org/wiki/Xorshift

    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}