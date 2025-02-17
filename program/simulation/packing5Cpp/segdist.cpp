#include "pch.h"
#include "segdist.h"
#include "potential.h"
#include <immintrin.h>
#include <cmath>

static inline __m128 _mm_absf(__m128 a) {
    // Create a mask with all bits set except the sign bit
    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    // Clear the sign bit of each float in the vector
    __m128 result = _mm_and_ps(a, mask);
    return result;
}

static inline float min4(float a[4]) {
    return fminf(fminf(a[0], a[1]), fminf(a[2], a[3]));
}

static bool isSegmentIntersect(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    float c1 = cosf(t1), s1 = sinf(t1), c2 = cosf(t2), s2 = sinf(t2);
    float x12 = x1 - x2, y12 = y1 - y2;
    // rectangle intersection
    if (abs(x12) < r * (abs(c1) + abs(c2)) && abs(y12) < r * (abs(s1) + abs(s2)))return true;
    // line intersection
    float rs = abs(r * sinf(t1 - t2)),
        xsyc1 = abs(x12 * s1 - y12 * c1),
        xsyc2 = abs(x12 * s2 - y12 * c2);
    return xsyc1 <= rs || xsyc2 <= rs;
}

static float segmentDistNoIntersect(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    float c1 = cosf(t1), s1 = sinf(t1), c2 = cosf(t2), s2 = sinf(t2);

    // calculate the distance between four pairs of points
    float ABCD[8], CDBA[8], AC_BD_CB_DA[4];
    float dx1 = r * c1, dy1 = r * s1, dx2 = r * c2, dy2 = r * s2;
    ABCD[0] = CDBA[6] = x1 - dx1; ABCD[2] = CDBA[4] = x1 + dx1; 
    ABCD[1] = CDBA[7] = y1 - dy1; ABCD[3] = CDBA[5] = y1 + dy1; 
    ABCD[4] = CDBA[0] = x2 - dx2; ABCD[6] = CDBA[2] = x2 + dx2;
    ABCD[5] = CDBA[1] = y2 - dy2; ABCD[7] = CDBA[3] = y2 + dy2;
    __m256 ABCDv = _mm256_loadu_ps(ABCD), CDBAv = _mm256_loadu_ps(CDBA);
    __m256 point_sub = _mm256_sub_ps(ABCDv, CDBAv);
    __m256 point_mul = _mm256_mul_ps(point_sub, point_sub);
    __m128 point_sum = _mm256_extractf128_ps(_mm256_hadd_ps(point_mul, point_mul), 0);
    _mm_store_ps(AC_BD_CB_DA, _mm_hadd_ps(point_sum, point_sum));
    float min1_sq = min4(AC_BD_CB_DA);

    // calculate the distance between point and line
    // P: midpoint of AB, Q: midpoint of CD
    float QQPP[8] = { x2, y2, x2, y2, x1, y1, x1, y1 };
    float sc[8] = { s1, c1, s1, c1, s2, c2, s2, c2 };
    float Acd_Bcd_Cab_Dab[4];
    // QA_QB_PC_PD = [xA - x2, yA - y2, xB - x2, yB - y2, xC - x1, yC - y1, xD - x1, yD - y1]
    __m256 QA_QB_PC_PD = _mm256_sub_ps(ABCDv, _mm256_loadu_ps(QQPP));
    __m256 QA_QB_PC_PD_mul = _mm256_mul_ps(QA_QB_PC_PD, _mm256_loadu_ps(sc));
    __m128 QA_QB_PC_PD_sub = _mm_absf(_mm256_extractf128_ps(_mm256_hsub_ps(QA_QB_PC_PD_mul, QA_QB_PC_PD_mul), 0));
    _mm_store_ps(Acd_Bcd_Cab_Dab, QA_QB_PC_PD_sub);
    float min2 = min4(Acd_Bcd_Cab_Dab);

    return fminf(sqrtf(min1_sq), min2);
}

float SegDist(float r, float x1, float y1, float t1, float x2, float y2, float t2)
{
    if (isSegmentIntersect(r, x1, y1, t1, x2, y2, t2))return 0.0f;
    return segmentDistNoIntersect(r, x1, y1, t1, x2, y2, t2);
}
