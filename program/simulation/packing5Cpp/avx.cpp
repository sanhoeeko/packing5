#include "pch.h"
#include <algorithm>
#include <immintrin.h>
#include <random>

void SumTensor4(void* p_z, void* p_Gij, void* p_gi, int N)
{
    int* z = (int*)p_z;
    float* Gij = (float*)p_Gij;
    float* gi = (float*)p_gi;

    for (int i = 0; i < N; ++i) {
        __m128 sum = _mm_loadu_ps(Gij + 4 * ((i + 1) * max_neighbors - 1));
        float* ptr_start = Gij + 4 * i * max_neighbors;
        int j_max = 4 * z[i];
        for (int j = 0; j < j_max; j += 4) {
            __m128 element = _mm_loadu_ps(ptr_start + j);
            sum = _mm_add_ps(sum, element);
        }
        _mm_storeu_ps(gi + i * 4, sum);
    }
}

void HollowClear(void* p_i32, int N, int stride) {
    int* p = (int*)p_i32;
    for (int i = 0; i < N; i++) {
        p[i * stride] = 0;
    }
}

#if simd_256_bits

void AddVector4(void* p_x, void* p_g, int N, float s) {
    /*
        require: number of particles being a multiple of 4
    */
    float* x = (float*)p_x;
    float* g = (float*)p_g;
    __m256 s_vec = _mm256_set1_ps(s);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 x_vec1 = _mm256_loadu_ps(x + i8);
        __m256 g_vec1 = _mm256_loadu_ps(g + i8);
        __m256 result_vec1 = _mm256_add_ps(x_vec1, _mm256_mul_ps(g_vec1, s_vec));
        _mm256_storeu_ps(x + i8, result_vec1);
    }
}

void PerturbVector4(void* p_input, int N, float sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, sigma);
    float* g = (float*)p_input;

    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 g_vec8 = _mm256_loadu_ps(g + i8);
        __m256 rand_vec8 = _mm256_setr_ps(
            dis(gen), dis(gen), dis(gen), 0,
            dis(gen), dis(gen), dis(gen), 0
        );
        __m256 result_vec = _mm256_add_ps(g_vec8, rand_vec8);
        _mm256_storeu_ps(g + i8, result_vec);
    }
}

void FastClear(void* p_float, int size) {
    float* ptr = (float*)p_float;
    __m256 zero_vec = _mm256_setzero_ps();

    for (int i8 = 0; i8 < size; i8 += 8) {
        _mm256_storeu_ps(ptr + i8, zero_vec);
    }
}

float FastNorm(void* p_x, int n) {
    float* x = (float*)p_x;
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 vec = _mm256_loadu_ps(x + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vec, vec));
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    return std::sqrt(result[0] + result[1] + result[2] + result[3] +
        result[4] + result[5] + result[6] + result[7]);
}

#else

void AddVector4(void* p_x, void* p_g, int N, float s)
{
    float* x = (float*)p_x;
    float* g = (float*)p_g;
    for (int i4 =0; i4 < N * 4; i4 += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i4);
        __m128 g_vec = _mm_loadu_ps(g + i4);
        __m128 s_vec = _mm_set1_ps(s);
        __m128 sg_vec = _mm_mul_ps(g_vec, s_vec);
        __m128 result_vec = _mm_add_ps(x_vec, sg_vec);
        _mm_storeu_ps(x + i4, result_vec);
    }
}

void PerturbVector4(void* p_input, int N, float sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, sigma);
    float* g = (float*)p_input;

    for (int i4 = 0; i4 < N * 4; i4 += 4) {
        __m128 g_vec4 = _mm_loadu_ps(g + i4);
        __m128 rand_vec4 = _mm_setr_ps(dis(gen), dis(gen), dis(gen), 0);
        __m128 result_vec = _mm_add_ps(g_vec4, rand_vec4);
        _mm_storeu_ps(g + i4, result_vec);
    }
}

void FastClear(void* p_float, int size) {
    float* ptr = (float*)p_float;
    __m128 zero_vec = _mm_setzero_ps();

    for (int i4 = 0; i4 < size; i4 += 4) {
        _mm_storeu_ps(ptr + i4, zero_vec);
    }
}

float FastNorm(void* p_x, int n) {
    float* x = (float*)p_x;
    __m128 sum = _mm_setzero_ps(); 
    for (int i = 0; i < n; i += 4) {
        __m128 vec = _mm_loadu_ps(x + i); 
        sum = _mm_add_ps(sum, _mm_mul_ps(vec, vec)); 
    } 
    float result[4]; 
    _mm_storeu_ps(result, sum); 
    return std::sqrt(result[0] + result[1] + result[2] + result[3]); 
}

#endif