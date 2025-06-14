#include "pch.h"
#include "potential.h"
#include "functional.h"
#include "array.h"
#include "segdist.h"
#include <math.h>
#include <vector>
#include <omp.h>

using namespace std;

float modpi(float x)
{
    const float a = 1 / pi;
    float y = x * a;
    return y - floor(y);
}

template<int capacity>
static inline int hashFloat2Pi(const float& x) {
    /*
        using "bitwise and" for fast modulo. 
        require: `capacity` is a power of 2.
    */
    const float a = capacity / (2 * pi);
    const int mask = capacity - 1;
    return (int)(a * x) & mask;
}
template<int capacity>
static inline int hash04(const float& x) {
    return x * (capacity / 4);
}

template<>
int anyHasher<float, _h2pi>(const float& x) {
    return hashFloat2Pi<sz1d>(x);
}
template<>
int anyHasher<float, _h4>(const float& x) {
    return hash04<sz1d>(x);
}

float scalar_f(float* scalar_func_of_r2, float r2) {
    return r2 >= 4 ? 0 : scalar_func_of_r2[hash04<sz1d>(r2)];
}

static inline float _sin(const float& x) { return sin(x); }
static inline float _cos(const float& x) { return cos(x); }
static inline float _hertzianSq(const float& x2) { return pow(2 - sqrt(x2), 2.5f); }
static inline float _hertzianSqDR(const float& x2) {
    float x = sqrt(x2);
    if (x == 0)return 0;
    return 2.5 * pow(2 - x, 1.5f) / x;
}

static LookupFunc<float, float, sz1d, _h2pi> FSin() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h2pi>(_sin, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h2pi> FCos() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h2pi>(_cos, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h4> FHertzianSq() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_hertzianSq, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h4> FHertzianSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_hertzianSqDR, xs);
    return *f;
}

float fsin(float x) { 
    static auto _fsin = FSin();
    return _fsin(x); 
}

float fcos(float x) { 
    static auto _fcos = FCos();
    return _fcos(x); 
}

float fHzSq(float x2) {
    static auto _fhz = FHertzianSq();
    return _fhz(x2);
}

float fHzSqDR(float x2) {
    static auto _fhz = FHertzianSqDR();
    return _fhz(x2);
}


xyt ParticleShape::transform(const xyt& q)
{
    return
    {
        abs(q.x) / (2 * a_padded),
        abs(q.y) / (a_padded + b_padded),
        (q.x > 0) ^ (q.y > 0) ? modpi(q.t) : 1 - modpi(q.t),
        q.unused
    };
}

xyt ParticleShape::transform_signed(const xyt& g)
{
    return
    {
        g.x / (2 * a_padded),
        g.y / (a_padded + b_padded),
        g.t / pi,
        g.unused
    };
}

xyt ParticleShape::inverse(const xyt& q)
{
    return
    {
        (2 * a_padded) * q.x,
        (a_padded + b_padded) * q.y,
        pi * q.t,
        q.unused
    };
}

bool ParticleShape::isSegmentCrossing(const xyt& q)
{
    return
        q.y < c * fsin(q.t) &&
        q.y * fcos(q.t) >(q.x - c) * fsin(q.t);
}

inline static XytPair ZeroXytPair() {
    return { 0,0,0,0,0,0 };
}

void ParticleShape::initPotential(int threads, float* scalar_potential) {
    vector<float> xs = linspace_including_endpoint(0, 1, szx);
    vector<float> ys = linspace_including_endpoint(0, 1, szy);
    vector<float> ts = linspace_including_endpoint(0, 1, szt);
    int m1 = xs.size();
    int m2 = ys.size();
    int m3 = ts.size();
    omp_set_num_threads(threads); 

    // use all cores; collapse loops for better parallelization
#pragma omp parallel for collapse(1)
    for (int i = 0; i < m1; i++) {
        float x = xs[i];
        for (int j = 0; j < m2; j++) {
            float y = ys[j];
            for (int k = 0; k < m3; k++) {
                float t = ts[k];
                xyt q = { x,y,t };
                data[i][j][k] = StandardPotential(inverse(q), scalar_potential);
            }
        }
    }
}

xyt ParticleShape::potential(const xyt& q) {
    /*
        q: real x y theta
    */
    xyt g = transform_signed(interpolatePotential(transform(q)));
    bool
        sign_x = q.x > 0,
        sign_y = q.y > 0;
    if (sign_x)g.x = -g.x;
    if (sign_y)g.y = -g.y;
    if (!(sign_x ^ sign_y))g.t = -g.t;
    return g;
}

xyt ParticleShape::gradient(const xyt& q) {
    /*
        q: real x y theta
    */
    xyt g = transform_signed(interpolateGradient(transform(q)));
    bool
        sign_x = q.x > 0,
        sign_y = q.y > 0;
    if (sign_x)g.x = -g.x;
    if (sign_y)g.y = -g.y;
    if (!(sign_x ^ sign_y))g.t = -g.t;
    return g;
}

struct RotVector
{
    float s, c;

    RotVector(float angle) {
        s = fsin(angle); c = fcos(angle);
    }

    void rot(float* ptr, float* dst) {
        float x = ptr[0], y = ptr[1];
        dst[0] = c * x - s * y;
        dst[1] = s * x + c * y;
    }
    void inv(float* ptr, float* dst) {
        float x = ptr[0], y = ptr[1];
        dst[0] = c * x + s * y;
        dst[1] = -s * x + c * y;
    }
};

inline static float crossProduct(float* r, float* f) {
    return r[0] * f[1] - r[1] * f[0];
}

template<>
GePair singleGradient<Normal>(Rod* shape, float x, float y, float t1, float t2) {
    xyt input = { x,y };
    xyt temp;
    RotVector rv = RotVector(t2);
    rv.inv(&input.x, &temp.x);
    temp.t = t2 - t1;
    xyt gradient = shape->gradient(temp);
    rv.rot((float*)&gradient, (float*)&gradient);
    float moment2 = -crossProduct(&input.x, (float*)&gradient) - gradient.t;   // parallel axis theorem !!
    return { gradient, {-gradient.x, -gradient.y, moment2} };
}

template<>
GePair singleGradient<AsDisks>(Rod* shape, float x, float y, float t1, float t2) {
    float r2 = x * x + y * y;
    float fr = fHzSqDR(r2);
    float fx = fr * x, fy = fr * y;
    return { {fx, fy, 0}, {-fx, -fy, 0} };
}

template<>
GePair singleGradientAndEnergy<Normal>(Rod* shape, float x, float y, float t1, float t2) {
    xyt input = { x,y };
    xyt temp;
    RotVector rv = RotVector(t2);
    rv.inv(&input.x, &temp.x);
    temp.t = t2 - t1;
    xyt gradient = shape->potential(temp);
    rv.rot((float*)&gradient, (float*)&gradient);
    float moment2 = -crossProduct(&input.x, (float*)&gradient) - gradient.t;   // parallel axis theorem !!
    return { gradient, {-gradient.x, -gradient.y, moment2} };
}

Rod::Rod(int n, float d, float (*data_ptr)[szy][szt]) {
    this->data = data_ptr;
    a = 1;
    b = 1 / (1 + (n - 1) * d / 2.0f);
    c = a - b;
    this->n = n;
    this->rod_d = d * b;
    a_padded = a + 0.01f;    // zero padding, for memory safety
    b_padded = b + 0.01f;
    this->n_shift = -(n - 1) / 2.0f;
    this->inv_disk_R2 = 1 / (b * b);
}

/*
    The origin definition of the potential
*/
float Rod::StandardPotential(const xyt& q, float* scalar_potential) {
    float v = 0;
    for (int k = 0; k < n; k++) {
        float z1 = (n_shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float z2 = (n_shift + l) * rod_d;
            float
                xij = q.x - z1 + z2 * fcos(q.t),
                yij = q.y + z2 * fsin(q.t),
                r2 = (xij * xij + yij * yij) * inv_disk_R2;
            v += scalar_f(scalar_potential, r2);
        }
    }
    return v;
}

XytPair Rod::StandardGradient(float x, float y, float t1, float t2, float* scalar_potential_dr) {
    XytPair g = ZeroXytPair();
    for (int k = 0; k < n; k++) {
        float z1 = (n_shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float z2 = (n_shift + l) * rod_d;
            float
                xij = x - z1 * fcos(t1) + z2 * fcos(t2),
                yij = y - z1 * fsin(t1) + z2 * fsin(t2),
                r2 = (xij * xij + yij * yij) * inv_disk_R2,
                fr = scalar_f(scalar_potential_dr, r2);
            if (fr != 0.0f) {
                float
                    fx = fr * xij,
                    fy = fr * yij,
                    torque1 = z1 * (fx * fsin(t1) - fy * fcos(t1)),
                    torque2 = -z2 * (fx * fsin(t2) - fy * fcos(t2));
                g.first += {fx, fy, torque1};
                g.second += {-fx, -fy, torque2};
            }
        }
    }
    return g;
}

Segment::Segment(float gamma, float (*data_ptr)[szy][szt]) {
    this->data = data_ptr;
    this->r = 1 - 1 / gamma;
    a = 1;
    b = 1 / gamma;
    c = a - b;
    a_padded = a + 0.01f;
    b_padded = b + 0.01f;
}

float Segment::StandardPotential(const xyt& q, float* scalar_potential)
{
    float h = SegDist(r, q.x, q.y, q.t, 0, 0, 0) / b;       // (h / b) is in [0, 2]
    return scalar_f(scalar_potential, h * h);
}
