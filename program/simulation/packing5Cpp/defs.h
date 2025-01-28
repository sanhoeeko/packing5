#pragma once

#define max_neighbors 12
#define LBFGS_MEMORY_SIZE 20

#define DIGIT_X 9
#define DIGIT_Y 9
#define DIGIT_T 9
#define DIGIT_R 20
#define ENERGY_STRIDE 1000

#include <omp.h>
#include <cmath>
#include <iostream>
using namespace std;
#undef max
#undef min

const float pi = 3.141592654;
const float max_gradient_amp = 100;

enum HowToCalGradient { Normal, AsDisks, HowToCalGradient_Count };
enum HashFunc { _h2pi, _h4 };

struct xyt {
    float x, y, t, unused;
    void operator+=(const xyt&); void operator-=(const xyt&);
    xyt operator*(const float);
};
typedef xyt ge;                             // struct ge { float gx, gy, gt, E; };
struct XytPair { xyt first, second; };
typedef XytPair GePair;
