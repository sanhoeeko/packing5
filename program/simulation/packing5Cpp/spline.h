#pragma once

#include <array>
#include <immintrin.h>

/*
    Four auxiliary functions defined in [0,1]
*/
template<int N> inline float f(float t);
template<> inline float f<1>(float t) { return t * t * t; }
template<> inline float f<2>(float t) { return 1 + 3 * (t + t * t - t * t * t); }
template<> inline float f<3>(float t) { return f<2>(1 - t); }
template<> inline float f<4>(float t) { return f<1>(1 - t); }

/*
    Four derivatives defined in [0,1]
*/
template<int N> inline float g(float t);
template<> inline float g<1>(float t) { return 3 * t * t; }
template<> inline float g<2>(float t) { return 3 * (1 + 2 * t - 3 * t * t); }
template<> inline float g<3>(float t) { return -g<2>(1 - t); }
template<> inline float g<4>(float t) { return -g<1>(1 - t); }

template<int I, int J, int K>
constexpr __m128 _GEijk(float dx, float dy, float dt, float C[4][4][4]) {
    float GxytE[4] = {
        g<I>(dx)* f<J>(dy)* f<K>(dt),
        f<I>(dx)* g<J>(dy)* f<K>(dt),
        f<I>(dx)* f<J>(dy)* g<K>(dt),
        f<I>(dx)* f<J>(dy)* f<K>(dt),
    };
    __m128 c = _mm_set_ps1(C[I][J][K]);
    return _mm_mul_ps(c, _mm_loadu_ps(GxytE));
}

template<int... Indices>
inline __m128 _sumGE(float dx, float dy, float dt, const float(&C)[4][4][4], std::integer_sequence<int, Indices...>) {
    return _mm_add_ps((_GEijk<Indices / 16, (Indices / 4) % 4, Indices % 4>(dx, dy, dt, C)), ...);
}