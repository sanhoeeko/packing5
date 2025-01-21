#pragma once

using namespace std;

struct v4 {
    int N;
    bool cited;
    float* data;

    v4();
    v4(int N);
    v4(int N, float* data_ptr);
    void die();
    float dot(const v4& y);
    void set(float* src);

    void equals_sub(const v4& x, const v4& y);
    void operator+=(const v4& y);
    void operator-=(const v4& y);
    void operator*=(float s);

    void lbfgs_alg_1(float a, const v4& y);
    void lbfgs_alg_2(float a_b, const v4& s);
};

template<typename ty, int m>
struct RollList {
    ty data[m];

    RollList() { ; }

    ty& operator[](int idx) {
        return data[idx % m];
    }
};

template<int m>
struct RollList<v4, m> {
    v4 data[m];

    RollList(int N) {
        for (int i = 0; i < m; ++i) {
            data[i] = v4(N);
        }
    }
    ~RollList() {
        for (int i = 0; i < m; ++i) {
            data[i].die();
        }
    }
    v4& operator[](int idx) {
        return data[idx % m];
    }
};

template<int m>
struct L_bfgs {
    int N;
    int k;
    float* configuration_src;
    float* gradient_src;

    RollList<v4, m> x, g, s, y;
    RollList<float, m> a, b, rho;

    L_bfgs(int N, float* configuration_src, float* gradient_src);
    void init(float initial_stepsize);
    void update();
    void calDirection_to(v4& dst);
};

template<int m>
inline L_bfgs<m>::L_bfgs(int N, float* configuration_src, float* gradient_src)
    : x(N), g(N), s(N), y(N)
{
    this->N = N;
    this->k = 0;
    this->gradient_src = gradient_src;
    this->configuration_src = configuration_src;
}

template<int m>
inline void L_bfgs<m>::init(float initial_stepsize)
{
    k = 0;
    a[0] = initial_stepsize;
    x[0].set(configuration_src);
    g[0].set(gradient_src);
}

template<int m>
inline void L_bfgs<m>::update()
{
    const float e = 1e-6f;  // to avoid the denominator being zero
    x[k + 1].set(configuration_src);
    s[k].equals_sub(x[k + 1], x[k]);  // s[k] = x[k + 1] - x[k]; 
    g[k + 1].set(gradient_src);
    y[k].equals_sub(g[k + 1], g[k]);  // y[k] = g[k + 1] - g[k]; 
    rho[k] = 1.0f / (y[k].dot(s[k]) + e);
    k += 1;
}

template<int m>
inline void L_bfgs<m>::calDirection_to(v4& dst)
{
    /*
        This algorithm is from https://en.wikipedia.org/wiki/Limited-memory_BFGS
    */
    const float e = 1e-6f;  // to avoid the denominator being zero
    if (k < m) {
        dst.set(gradient_src);
    }
    else {
        // two-loop recursion
        // solve for the descent direction: z
        v4 z(N); z.set(gradient_src);
        for (int i = k - 1; i >= k - m; i--) {
            a[i] = rho[i] * s[i].dot(z);
            z.lbfgs_alg_1(a[i], y[i]);              // z -= a[i] * y[i];
        }
        z *= s[k - 1].dot(y[k - 1]) / (y[k - 1].dot(y[k - 1]) + e);
        for (int i = k - m; i <= k - 1; i++) {
            b[i] = rho[i] * y[i].dot(z);
            z.lbfgs_alg_2(a[i] - b[i], s[i]);        // z += (a[i] - b[i]) * s[i];
        }
        z *= -1;
        dst.set(z.data);
        z.die();
    }
}