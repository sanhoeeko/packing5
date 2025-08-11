#pragma once

#undef max
#undef min

#include <iostream>
#include <tuple>
using namespace std;

struct Point { float x, y; };
struct xyt3f { float x, y, t; };

const float pi = 3.141592654;

// If the number of particles is a multiple of 256, set it to "true".
// The number of particles must be a multiple of 8.
#define USE_AVX2 false
