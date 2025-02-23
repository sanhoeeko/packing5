#include "pch.h"
#include "segdist.h"
#include <immintrin.h>
#include <cmath>

static inline __m128 _mm_absf(__m128 a) {
    // Create a mask with all bits set except the sign bit
    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    // Clear the sign bit of each float in the vector
    __m128 result = _mm_and_ps(a, mask);
    return result;
}

static inline __m128 _mm_contract(__m256 input) {
    __m128 low = _mm256_castps256_ps128(input);
    __m128 high = _mm256_extractf128_ps(input, 1);
    return _mm_add_ps(low, high);
}

static inline __m128 _mm_contract_sub(__m256 input) {
    __m128 low = _mm256_castps256_ps128(input);
    __m128 high = _mm256_extractf128_ps(input, 1);
    return _mm_sub_ps(low, high);
}

static inline float a0b1c(float x, float a, float b, float c) {
    if (x < -1)return a;
    if (x > 1)return c;
    return b;
}

static inline float min4(float a[4]) {
    return fminf(fminf(a[0], a[1]), fminf(a[2], a[3]));
}

static bool isSegmentIntersect(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    float c1 = cosf(t1), s1 = sinf(t1), c2 = cosf(t2), s2 = sinf(t2);
    float x12 = x1 - x2, y12 = y1 - y2;
    // rectangle intersection
    if (abs(x12) >= r * (abs(c1) + abs(c2)) || abs(y12) >= r * (abs(s1) + abs(s2)))return false;
    // line intersection
    float rs = abs(r * sinf(t1 - t2)),
        xsyc1 = abs(x12 * s1 - y12 * c1),
        xsyc2 = abs(x12 * s2 - y12 * c2);
    return xsyc1 <= rs || xsyc2 <= rs;
}

static float segmentDistNoIntersect(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    // SIMD format: [x, x, x, x, y, y, y, y]
    float c1 = cosf(t1), s1 = sinf(t1), c2 = cosf(t2), s2 = sinf(t2);

    // calculate the distance between four pairs of points
    float ABCD[8], CDBA[8], AC_BD_CB_DA[4];
    float dx1 = r * c1, dy1 = r * s1, dx2 = r * c2, dy2 = r * s2;
    ABCD[0] = CDBA[3] = x1 - dx1; ABCD[4] = CDBA[7] = y1 - dy1;
    ABCD[1] = CDBA[2] = x1 + dx1; ABCD[5] = CDBA[6] = y1 + dy1;
    ABCD[2] = CDBA[0] = x2 - dx2; ABCD[6] = CDBA[4] = y2 - dy2;
    ABCD[3] = CDBA[1] = x2 + dx2; ABCD[7] = CDBA[5] = y2 + dy2;
    __m256 ABCDv = _mm256_loadu_ps(ABCD), CDBAv = _mm256_loadu_ps(CDBA);
    __m256 point_sub = _mm256_sub_ps(ABCDv, CDBAv);
    __m256 point_mul = _mm256_mul_ps(point_sub, point_sub);
    __m128 point_sum = _mm_contract(point_mul);
    __m128 point_dists = _mm_sqrt_ps(point_sum);
    _mm_store_ps(AC_BD_CB_DA, point_dists);

    // calculate the distance between point and line
    // P: midpoint of AB, Q: midpoint of CD
    float QQPP[8] = { x2, x2, x1, x1, y2, y2, y1, y1 };
    float sc[8] = { s2, s2, s1, s1, c2, c2, c1, c1 };
    float Acd_Bcd_Cab_Dab[4];
    // QA_QB_PC_PD = [xA - x2, xB - x2, xC - x1, xD - x1, yA - y2,  yB - y2,  yC - y1,  yD - y1]
    __m256 QA_QB_PC_PD = _mm256_sub_ps(ABCDv, _mm256_loadu_ps(QQPP));
    __m256 QA_QB_PC_PD_mul = _mm256_mul_ps(QA_QB_PC_PD, _mm256_loadu_ps(sc));
    __m128 QA_QB_PC_PD_sub = _mm_absf(_mm_contract_sub(QA_QB_PC_PD_mul));
    _mm_store_ps(Acd_Bcd_Cab_Dab, QA_QB_PC_PD_sub);

    // calculate the position parameter of points projected on the line
    // position parameter(U, AB) = (AU dot AB) / (AB dot AB)
    // here AB = (2r cos(t), 2r sin(t)), so (AU dot AB) / (AB dot AB) = (AU dot [cos(t), sin(t)]) / 2r
    float SC[8] = { c2, c2, c1, c1, s2, s2, s1, s1 };
    __m256 SCv = _mm256_loadu_ps(SC);
    __m256 vecs_mul = _mm256_mul_ps(QA_QB_PC_PD, SCv);
    __m128 vecs_sum = _mm_contract(vecs_mul);
    vecs_sum = _mm_mul_ps(vecs_sum, _mm_set_ps1(1 / r));
    float parameter_Acd_Bdc_Cba_Dab[4];
    _mm_store_ps(parameter_Acd_Bdc_Cba_Dab, vecs_sum);

    // determine the minimum according to the parameters
    float dists[4] = {
        a0b1c(parameter_Acd_Bdc_Cba_Dab[0], AC_BD_CB_DA[0], Acd_Bcd_Cab_Dab[0], AC_BD_CB_DA[3]),
        a0b1c(parameter_Acd_Bdc_Cba_Dab[1], AC_BD_CB_DA[2], Acd_Bcd_Cab_Dab[1], AC_BD_CB_DA[1]),
        a0b1c(parameter_Acd_Bdc_Cba_Dab[2], AC_BD_CB_DA[0], Acd_Bcd_Cab_Dab[2], AC_BD_CB_DA[2]),
        a0b1c(parameter_Acd_Bdc_Cba_Dab[3], AC_BD_CB_DA[3], Acd_Bcd_Cab_Dab[3], AC_BD_CB_DA[1]),
    };
    return min4(dists);
}

float SegDist(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    /*
        If two segments do not intersect, the distance is
        min(dist(A, CD), dist(B, CD), dist(C, AB), dist(D, AB))
    */
    if (isSegmentIntersect(r, x1, y1, t1, x2, y2, t2))return 0.0f;
    return segmentDistNoIntersect(r, x1, y1, t1, x2, y2, t2);
}
