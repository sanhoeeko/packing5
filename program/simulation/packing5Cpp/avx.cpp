#include "pch.h"
#include <algorithm>
#include <immintrin.h>
#include <random>
#include <string.h>
#include "lbfgs.h"
#include "myrand.h"

void SumTensor4(void* p_Gij, void* p_gi, int N)
{
    ge* Gij = (ge*)p_Gij;
    ge* gi = (ge*)p_gi;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < cores; j++) {
            gi[i] += Gij[i * cores + j];
        }
    }
}

void HollowClear(void* p_i32, int N, int stride) {
    int* p = (int*)p_i32;
    for (int i = 0; i < N; i++) {
        p[i * stride] = 0;
    }
}

void AddVector4(void* p_x, void* p_g, void* p_dst, int N, float s) {
    /*
        require: number of particles being a multiple of 4
    */
    float* x = (float*)p_x;
    float* g = (float*)p_g;
    float* dst = (float*)p_dst;
    __m256 s_vec = _mm256_set1_ps(s);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 x_vec1 = _mm256_loadu_ps(x + i8);
        __m256 g_vec1 = _mm256_loadu_ps(g + i8);
        __m256 result_vec1 = _mm256_add_ps(x_vec1, _mm256_mul_ps(g_vec1, s_vec));
        _mm256_storeu_ps(dst + i8, result_vec1);
    }
}

void PerturbVector4(void* p_input, int N, float sigma) {
    std::random_device true_random;
    xorshift32 gen(true_random());
    float* g = (float*)p_input;

    __m256 sig_vec = _mm256_set1_ps(sigma);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 g_vec8 = _mm256_loadu_ps(g + i8);
        __m256 rand_vec8 = _mm256_setr_ps(
            fast_gaussian(gen()), fast_gaussian(gen()), fast_gaussian(gen()), 0,
            fast_gaussian(gen()), fast_gaussian(gen()), fast_gaussian(gen()), 0
        );
        __m256 result_vec = _mm256_add_ps(g_vec8, _mm256_mul_ps(sig_vec, rand_vec8));
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
    return std::sqrt(result[0] + result[1] + result[2] /* + result[3]*/ +
        result[4] + result[5] + result[6] /* + result[7]*/ );
}

float MaxAbsVector4(void* p_x, int N)
{
    /* This is not an avx function. */
    float* x = (float*)p_x;
    float current_max = 0;
    for (int i = 0; i < 4*N; i+=4) {
        float absg = x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 1] * x[i + 2];
        current_max = absg > current_max ? absg : current_max;
    }
    return current_max;
}

void CwiseMulVector4(void* p_g, int N, float s) {
    /*
        require: number of particles being a multiple of 4
    */
    float* g = (float*)p_g;
    __m256 s_vec = _mm256_set1_ps(s);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 g_vec1 = _mm256_loadu_ps(g + i8);
        __m256 result_vec1 = _mm256_mul_ps(g_vec1, s_vec);
        _mm256_storeu_ps(g + i8, result_vec1);
    }
}

float DotVector4(void* p_a, void* p_b, int N) {
    /*
        require: number of particles being a multiple of 4
        return: a dot b
    */
    float* a = (float*)p_a;
    float* b = (float*)p_b;
    __m256 sum = _mm256_setzero_ps();
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i8);
        __m256 b_vec = _mm256_loadu_ps(b + i8);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[1] + result[2] /* + result[3]*/ +
        result[4] + result[5] + result[6] /* + result[7]*/;
}

v4::v4()
{
}

v4::v4(int N)
{
    this->N = N;
    this->cited = false;
    this->data = (float*)malloc(N * 4 * sizeof(float));
}

v4::v4(int N, float* data_ptr)
{
    this->N = N;
    this->cited = true;
    this->data = data_ptr;
}

void v4::die()
{
    if (!cited)free(data);
}

float v4::dot(const v4& y)
{
    return DotVector4(data, y.data, N);
}

void v4::set(float* src)
{
    memcpy(data, src, 4 * N * sizeof(float));
}

void _sub_vector4_to(void* p_a, void* p_b, void* p_c, int N) {
    /*
        require: number of particles being a multiple of 4
    */
    float* a = (float*)p_a;
    float* b = (float*)p_b;
    float* c = (float*)p_c;
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 a_vec1 = _mm256_loadu_ps(a + i8);
        __m256 b_vec1 = _mm256_loadu_ps(b + i8);
        __m256 result_vec1 = _mm256_sub_ps(a_vec1, b_vec1);
        _mm256_storeu_ps(c + i8, result_vec1);
    }
}

void v4::equals_sub(const v4& x, const v4& y)
{
    _sub_vector4_to(x.data, y.data, this->data, this->N);
}

void _add_vector4_inplace(void* p_a, void* p_b, int N) {
    /*
        require: number of particles being a multiple of 4
    */
    float* a = (float*)p_a;
    float* b = (float*)p_b;
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 a_vec1 = _mm256_loadu_ps(a + i8);
        __m256 b_vec1 = _mm256_loadu_ps(b + i8);
        __m256 result_vec1 = _mm256_add_ps(a_vec1, b_vec1);
        _mm256_storeu_ps(a + i8, result_vec1);
    }
}

void v4::operator+=(const v4& y)
{
    _add_vector4_inplace(data, y.data, N);
}

void _sub_vector4_inplace(void* p_a, void* p_b, int N) {
    /*
        require: number of particles being a multiple of 4
    */
    float* a = (float*)p_a;
    float* b = (float*)p_b;
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 a_vec1 = _mm256_loadu_ps(a + i8);
        __m256 b_vec1 = _mm256_loadu_ps(b + i8);
        __m256 result_vec1 = _mm256_sub_ps(a_vec1, b_vec1);
        _mm256_storeu_ps(a + i8, result_vec1);
    }
}

void v4::operator-=(const v4& y)
{
    _sub_vector4_inplace(data, y.data, N);
}

void v4::operator*=(float s)
{
    CwiseMulVector4(data, N, s);
}

void v4::lbfgs_alg_1(float a, const v4& v_y)
{
    // Meaning: z -= a[i] * y[i];
    float* z = this->data;
    float* y = v_y.data;
    __m256 a_vec = _mm256_set1_ps(a);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 z_vec = _mm256_loadu_ps(z + i8);
        __m256 y_vec = _mm256_loadu_ps(y + i8);
        _mm256_storeu_ps(z + i8, _mm256_sub_ps(z_vec, _mm256_mul_ps(a_vec, y_vec)));
    }
}

void v4::lbfgs_alg_2(float a_b, const v4& v_s)
{
    // z += (a[i] - b[i]) * s[i];
    float* z = this->data;
    float* s = v_s.data;
    __m256 a_b_vec = _mm256_set1_ps(a_b);
    for (int i8 = 0; i8 < N * 4; i8 += 8) {
        __m256 z_vec = _mm256_loadu_ps(z + i8);
        __m256 s_vec = _mm256_loadu_ps(s + i8);
        _mm256_storeu_ps(z + i8, _mm256_add_ps(z_vec, _mm256_mul_ps(s_vec, a_b_vec)));
    }
}

void ClipGradient(void* q_ptr, int N) {
    xyt* q = (xyt*)q_ptr;
    for (int i = 0; i < N; i++) {
        float absg2 = q[i].x * q[i].x + q[i].y * q[i].y + q[i].t * q[i].t;
        if (absg2 > max_gradient_amp * max_gradient_amp) {
            float ratio = max_gradient_amp / sqrtf(absg2);
            float* ptr = (float*)&q[i];
            __m128 vec = _mm_loadu_ps(ptr);
            __m128 r = _mm_set_ps1(ratio);
            _mm_storer_ps(ptr, _mm_mul_ps(vec, r));
        }
    }
}
