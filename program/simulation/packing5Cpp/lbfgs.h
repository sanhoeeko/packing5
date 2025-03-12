#pragma once

#include <string.h>
using namespace std;

struct v4 {
    int N;
    bool cited;
    float* data;

    v4();
    v4(int N);
    v4(int N, float* data_ptr);
    void die();
    void set(float* src);

    void equals_sub(const v4& x, const v4& y);
    void operator+=(const v4& y);
    void operator-=(const v4& y);
    void operator*=(float s);
    void mul_add(const v4& y, float a);
    float dot(v4& y);
};

template <typename T, size_t Capacity>
struct Deque
{
    size_t start, end, _size;
    std::vector<T> data;

    Deque() : start(0), end(0), _size(0), data(Capacity) {}

    void push_back(const T& value) {
        if (_size == Capacity) {
            start = (start + 1) % Capacity;
        }
        else {
            ++_size;
        }
        data[end] = value;
        end = (end + 1) % Capacity;
    }

    T& operator[](size_t index) {
        return data[(start + index) % Capacity];
    }

    const T& operator[](size_t index) const {
        return data[(start + index) % Capacity];
    }

    void clear() {
        start = 0;
        end = 0;
        _size = 0;
    }

    size_t size() const {
        return _size;
    }

    T& back() {
        return data[(end == 0 ? Capacity : end) - 1];
    }

    const T& back() const {
        return data[(end == 0 ? Capacity : end) - 1];
    }
};

template<int m>
struct L_bfgs {
    int N;
    int Dim;
    float* configuration_src;
    float* gradient_src;

    Deque<v4, m> s_history;
    Deque<v4, m> y_history;
    v4 s, y;
    v4 x_prev, g_prev;

    L_bfgs(int N, float* configuration_src, float* gradient_src);
    void update(float* x_new, float* g_new);
    void calDirection_to(float* p);
};

template<int m>
inline L_bfgs<m>::L_bfgs(int N, float* configuration_src, float* gradient_src)
    : s(N), y(N), x_prev(N), g_prev(N), N(N), 
    configuration_src(configuration_src), gradient_src(gradient_src) 
{
    Dim = 4 * N;
}

template<int m>
inline void L_bfgs<m>::update(float* fp_x_new, float* fp_g_new)
{
    v4 x_new(N, fp_x_new);
    v4 g_new(N, fp_g_new);
    s.equals_sub(x_new, x_prev);
    y.equals_sub(g_new, g_prev);

    // numerical stability check
    float ys = y.dot(s);
    if (ys > 1e-10) {
        s_history.push_back(s);
        y_history.push_back(y);
    }
}

float DotVector4(void* p_a, void* p_b, int N);

template<int m>
inline void L_bfgs<m>::calDirection_to(float* p)
{
    /*
        This algorithm is from https://en.wikipedia.org/wiki/Limited-memory_BFGS
        Created by DeepSeek
    */

    //collect data for update
    x_prev.set(configuration_src);
    g_prev.set(gradient_src);

    if (s_history.size() < m) {
        // reduce to gradient descent if the history is unfilled
        memcpy(p, gradient_src, Dim * sizeof(float));
    }
    else {
        v4 q(N, p); q.set(gradient_src);
        std::vector<float> alphas(m);

        // 逆序处理历史向量
        for (int i = m - 1; i >= 0; --i) {
            float rho = 1.0 / y_history[i].dot(s_history[i]);
            alphas[i] = rho * q.dot(s_history[i]);
            q.mul_add(y_history[i], -alphas[i]);
        }

        // 计算Hessian缩放因子
        float gamma = s_history.back().dot(y_history.back()) / y_history.back().dot(y_history.back());
        q *= gamma;

        // 正序处理历史向量
        for (int i = 0; i < m; ++i) {
            float rho = 1.0 / y_history[i].dot(s_history[i]);
            float beta = rho * q.dot(y_history[i]);
            q.mul_add(s_history[i], alphas[i] - beta);
        }

        // 得到搜索方向
        // memcpy(p, q.data, Dim * sizeof(float));
    }

    // 强制下降方向检查
    if (DotVector4(p, gradient_src, N) >= 0) {
        memcpy(p, gradient_src, Dim * sizeof(float));
        s_history.clear();
        y_history.clear();
    }
}