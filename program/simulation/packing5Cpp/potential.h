#pragma once

#include"defs.h"
#include"functional.h"

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of int.
*/
const int szx = 1ll << DIGIT_X;
const int szy = 1ll << DIGIT_Y;
const int szt = 1ll << DIGIT_T;
const int sz1d = 1ll << DIGIT_R;
const int szxyt = szx * szy * szt;

float fsin(float x);
float fcos(float x);

float modpi(float x);

/*
    Base class. It is ndependent of the shape of anisotropic particles
*/
struct ParticleShape {
    float
        a, b, c,
        a_padded, b_padded;
    ParticleShape() { ; }
    xyt transform(const xyt& q);
    xyt transform_signed(const xyt& q);
    xyt inverse(const xyt& q);
    bool isSegmentCrossing(const xyt& q);
};

struct Rod : ParticleShape {
    float
        rod_d,
        n_shift,
        inv_disk_R2;
    int n;
    float (*data)[szy][szt];

    // initialization
    Rod(int n, float d, float (*data_ptr)[szy][szt]);
    void initPotential(int threads, float* scalar_potential);

    // original definitions
    float StandardPotential(const xyt& q, float* scalar_potential);
    XytPair StandardGradient(float x, float y, float t1, float t2, float* scalar_potential_dr);

    // auxiliary functions 
    xyt interpolateGradientSimplex(const xyt& q);
    xyt interpolatePotentialSimplex(const xyt& q);
    float potentialNoInterpolate(const xyt& q);

    // interfaces
    xyt gradient(const xyt& q);
    xyt potential(const xyt& q);
};

template<HowToCalGradient how>
GePair singleGradient(Rod* shape, float x, float y, float t1, float t2);

template<HowToCalGradient how>
GePair singleGradientAndEnergy(Rod* shape, float x, float y, float t1, float t2);

template<HowToCalGradient how, bool need_energy>
GePair singleGE(Rod* shape, float x, float y, float t1, float t2)
{
    if constexpr (need_energy) {
        return singleGradientAndEnergy<how>(shape, x, y, t1, t2);
    }
    else {
        return singleGradient<how>(shape, x, y, t1, t2);
    }
}