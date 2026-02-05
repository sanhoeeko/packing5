#include "pch.h"
#include "potential.h"

xyt ParticleShape::interpolateGradientSimplex(const xyt& q)
{
    if (q.y >= 1)return { 0,0,0,0 };
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 4 points:
        (i,j,k), (i �� 1,j,k), (i,j �� 1,k), (i,j,k �� 1)
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = round(X),
        j = round(Y),
        k = round(T);
    int
        hi = i <= X ? 1 : -1,   // do not use '<', because X and i can be both 0.0f and hi = -1 causes an illegal access
        hj = j <= Y ? 1 : -1,
        hk = k <= T ? 1 : -1;
    float
        v000 = data[i][j][k],
        v100 = data[i + hi][j][k],
        v010 = data[i][j + hj][k],
        v001 = data[i][j][k + hk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * a1 * hi,
        B = (-v000 + v010) * a2 * hj,
        C = (-v000 + v001) * a3 * hk;
    // D = v000;
/*
    the gradient: (A,B,C) is already obtained. (if only cauculate gradient, directly return)
    the value: A(x-x0) + B(y-y0) + C(t-t0) + D
*/
    return { A,B,C,0 };
}


xyt ParticleShape::interpolatePotentialSimplex(const xyt& q)
{
    if (q.y >= 1)return { 0,0,0,0 };
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 4 points:
        (i,j,k), (i �� 1,j,k), (i,j �� 1,k), (i,j,k �� 1)
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = round(X),
        j = round(Y),
        k = round(T);
    float
        dx = (X - i) / a1,
        dy = (Y - j) / a2,
        dt = (T - k) / a3;
    int
        hi = dx >= 0 ? 1 : -1,
        hj = dy >= 0 ? 1 : -1,
        hk = dt >= 0 ? 1 : -1;
    float
        v000 = data[i][j][k],
        v100 = data[i + hi][j][k],
        v010 = data[i][j + hj][k],
        v001 = data[i][j][k + hk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * a1 * hi,
        B = (-v000 + v010) * a2 * hj,
        C = (-v000 + v001) * a3 * hk,
        D = v000;
    /*
        the energy: A(x-x0) + B(y-y0) + C(t-t0) + D
        since x0 <- floor'(x), there must be x > x0, y > y0, t > t0
    */
    float energy = A * dx + B * dy + C * dt + D;
    return { A, B, C, energy };
}

xyt ParticleShape::interpolateGradientTrilinear(const xyt& q)
{
    if (q.y >= 1)return { 0,0,0,0 };
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 8 points
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = floor(X),
        j = floor(Y),
        k = floor(T);
    float  // dx, dy, dt in [0, 1]
        dx = X - i, bx = 1 - dx,
        dy = Y - j, by = 1 - dy,
        dt = T - k, bt = 1 - dt;
    float v[8];
    for (int hi = 0; hi <= 1; hi++) {
        for (int hj = 0; hj <= 1; hj++) {
            for (int hk = 0; hk <= 1; hk++) {
                v[(hi << 2) | (hj << 1) | hk] = data[i + hi][j + hj][k + hk];
            }
        }
    }
    // trilinear calculation
    float 
        Gx = (v[0b100] - v[0b000]) * by * bt + (v[0b101] - v[0b001]) * by * dt + (v[0b110] - v[0b010]) * dy * bt + (v[0b111] - v[0b011]) * dy * dt,
        Gy = (v[0b010] - v[0b000]) * bx * bt + (v[0b011] - v[0b001]) * bx * dt + (v[0b110] - v[0b100]) * dx * bt + (v[0b111] - v[0b101]) * dx * dt,
        Gt = (v[0b001] - v[0b000]) * bx * by + (v[0b011] - v[0b010]) * bx * dy + (v[0b101] - v[0b100]) * dx * by + (v[0b111] - v[0b110]) * dx * dy;
    return { Gx, Gy, Gt, 0 };
}

xyt ParticleShape::interpolatePotentialTrilinear(const xyt& q)
{
    if (q.y >= 1)return { 0,0,0,0 };
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 8 points
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = floor(X),
        j = floor(Y),
        k = floor(T);
    float  // dx, dy, dt in [0, 1]
        dx = X - i, bx = 1 - dx,
        dy = Y - j, by = 1 - dy,
        dt = T - k, bt = 1 - dt;
    float v[8];
    for (int hi = 0; hi <= 1; hi++) {
        for (int hj = 0; hj <= 1; hj++) {
            for (int hk = 0; hk <= 1; hk++) {
                v[(hi << 2) | (hj << 1) | hk] = data[i + hi][j + hj][k + hk];
            }
        }
    }
    // trilinear calculation
    float
        E = bx * (by * (v[0b000] * bt + v[0b001] * dt) + dy * (v[0b010] * bt + v[0b011] * dt))
          + dx * (by * (v[0b100] * bt + v[0b101] * dt) + dy * (v[0b110] * bt + v[0b111] * dt)),
        Gx = (v[0b100] - v[0b000]) * by * bt + (v[0b101] - v[0b001]) * by * dt + (v[0b110] - v[0b010]) * dy * bt + (v[0b111] - v[0b011]) * dy * dt,
        Gy = (v[0b010] - v[0b000]) * bx * bt + (v[0b011] - v[0b001]) * bx * dt + (v[0b110] - v[0b100]) * dx * bt + (v[0b111] - v[0b101]) * dx * dt,
        Gt = (v[0b001] - v[0b000]) * bx * by + (v[0b011] - v[0b010]) * bx * dy + (v[0b101] - v[0b100]) * dx * by + (v[0b111] - v[0b110]) * dx * dy;
    return { Gx, Gy, Gt, E };
}


// new SIMD method

__m128 set_element(__m128 vec, int index, float value) {
    // -1 = 0xfffffff
    alignas(16) static const int masks[4][4] = {
        {-1, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, -1}
    };

    __m128 mask = _mm_load_ps((const float*)masks[index]);
    __m128 v = _mm_set1_ps(value);

    // clear old value, set new value
    return _mm_or_ps(_mm_andnot_ps(mask, vec), _mm_and_ps(mask, v));
}

xyt ParticleShape::_trilinearInterpolate(const xyt& q_xyt)
{
    static bool alerted = false;
    if (!alerted) {
        cout << "Using SIMD trilinear interpolation for gradient calculation." << endl;
        alerted = true;
    }
    if (q_xyt.y >= 1)return { 0,0,0,0 };
    __m128 q = _mm_loadu_ps(&q_xyt.x);  // [q.x, q.y, q.t, unused]

    // const float a1 = szx - 1, a2 = szy - 1, a3 = szt - 1;
    const __m128 a = _mm_setr_ps(szx - 1.0f, szy - 1.0f, szt - 1.0f, 0.0f);

    // float X = q.x * a1, Y = q.y * a2, T = q.t * a3;
    __m128 q_scaled = _mm_mul_ps(q, a);

    // int i = floor(X), j = floor(Y), k = floor(T);
    __m128 q_floor = _mm_floor_ps(q_scaled);  // [i, j, k, 0]

    // float dx = X - i, dy = Y - j, dt = T - k;
    // Rescaled dx, dy, dt are in [0, 1]
    __m128 u_vec = _mm_sub_ps(q_scaled, q_floor);  // [u, v, w, 0]

    // float bx = 1 - dx, by = 1 - dy, bt = 1 - dt;
    __m128 one_minus_u_vec = _mm_sub_ps(_mm_set1_ps(1.0f), u_vec);  // [1-u, 1-v, 1-w, 1]

    // Extract individual values for data access  
    __m128i q_floor_int = _mm_cvtps_epi32(q_floor);
    int* ijk = (int*)&q_floor_int;
    int i = ijk[0], j = ijk[1], k = ijk[2];

    // Data (memory) access
    float f[8];
    __int64* double_v = (__int64*)f;

    //f[0b000] = data[i][j][k];
    //f[0b001] = data[i][j][k + 1];
    double_v[0b00] = *(__int64*)&data[i][j][k];
    //f[0b010] = data[i][j + 1][k];
    //f[0b011] = data[i][j + 1][k + 1];
    double_v[0b01] = *(__int64*)&data[i][j + 1][k];
    //f[0b100] = data[i + 1][j][k];
    //f[0b101] = data[i + 1][j][k + 1];
    double_v[0b10] = *(__int64*)&data[i + 1][j][k];
    //f[0b110] = data[i + 1][j + 1][k];
    //f[0b111] = data[i + 1][j + 1][k + 1];
    double_v[0b11] = *(__int64*)&data[i + 1][j + 1][k];

    __m128 vec_x0 = _mm_load_ps(f);
    __m128 vec_x1 = _mm_load_ps(f + 4);
    float* u = (float*)&u_vec;
    float* one_minus_u = (float*)&one_minus_u_vec;

    // interpolate along x-direction
    // M,, = (1-u) * f0,, + u * f1,, -> [M00, M01, M10, M11]
    __m128 M = _mm_add_ps(_mm_mul_ps(vec_x0, _mm_set1_ps(one_minus_u[0])), _mm_mul_ps(vec_x1, _mm_set1_ps(u[0])));
    // dM,,/du = f1,, - f0,,
    __m128 dM_du = _mm_sub_ps(vec_x1, vec_x0);

    // interpolate along y-direction
    // N, = (1-v) * M0, + v * M1,
    // dN,/dv = M1, - M0,
    // dN,/du = (1-v) * dM0,/du + v * dM1,/du
    // So, the vector [N0, N1, dN0/du, dN1/du] can be calculated by SIMD
    __m128 J = _mm_shuffle_ps(M, dM_du, _MM_SHUFFLE(1, 0, 1, 0));  // [M00, M01, dM00/du, dM01/du]
    __m128 K = _mm_shuffle_ps(M, dM_du, _MM_SHUFFLE(3, 2, 3, 2));  // [M10, M11, dM10/du, dM11/du]
    // [N0, N1, dN0/du, dN1/du]
    __m128 N_and_dN_du = _mm_add_ps(_mm_mul_ps(J, _mm_set1_ps(one_minus_u[1])), _mm_mul_ps(K, _mm_set1_ps(u[1])));

    float* m = (float*)&M;
    float dN0_dv = m[0b10] - m[0b00];
    float dN1_dv = m[0b11] - m[0b01];

    // interpolate along z-direction
    
    float* N = (float*)&N_and_dN_du;
    float N0 = N[0], N1 = N[1], dN0_du = N[2], dN1_du = N[3];
    float w0 = one_minus_u[2], w1 = u[2];
    float dF_du = w0 * dN0_du + w1 * dN1_du;
    float dF_dv = w0 * dN0_dv + w1 * dN1_dv;
    float dF_dw = N1 - N0;
    float F = w0 * N0 + w1 * N1;
    //return { dF_du, dF_dv, dF_dw, F };
    __m128 result = _mm_setr_ps(dF_du, dF_dv, dF_dw, F);
    return *(xyt*)&result;
    
    /*float* N = (float*)&N_and_dN_du;
    float N0 = N[0], N1 = N[1], dN0_du = N[2], dN1_du = N[3];
    float w0 = one_minus_u[2], w1 = u[2];
    __m128 R = _mm_setr_ps(dN0_du, dN0_dv, 0, N0);
    __m128 S = _mm_setr_ps(dN1_du, dN1_dv, 0, N1);
    __m128 result = _mm_add_ps(_mm_mul_ps(R, _mm_set1_ps(w0)), _mm_mul_ps(S, _mm_set1_ps(w1)));
    ((float*)&result)[2] = N1 - N0;

    return *(xyt*)&result;*/

    // dF/du = (1-w) * dN0/du + w * dN1/du
    // dF/dv = (1-w) * dN0/dv + w * dN1/dv
    // dF/dw = N1 - N0 (= (-1)*N0 + 1*N1)
    // F = (1-w) * N0 + w * N1

    // [dN0/du, dN0/dv, N0, N0]
    //__m128 R = _mm_shuffle_ps(N_and_dN_du, N_and_dN_du, _MM_SHUFFLE(0, 0, 2, 2));
    //R = set_element(R, 1, dN0_dv);
    //// [dN1/du, dN1/dv, N1, N1]
    //__m128 S = _mm_shuffle_ps(N_and_dN_du, N_and_dN_du, _MM_SHUFFLE(1, 1, 3, 3));
    //S = set_element(S, 1, dN1_dv);
    //// [1-w, 1-w, -1, 1-w]
    //// [w, w, 1, w]
    //__m128 one_minus_w_vec = _mm_set1_ps(one_minus_u[2]);
    //one_minus_w_vec = set_element(one_minus_w_vec, 2, -1.0f);
    //__m128 w_vec = _mm_set1_ps(u[2]);
    //w_vec = set_element(w_vec, 2, 1.0f);

    //// [dF/du, dF/dv, dF/dw, F]
    //__m128 result = _mm_add_ps(_mm_mul_ps(R, one_minus_w_vec), _mm_mul_ps(S, w_vec));

    //xyt result_xyt;
    //_mm_storeu_ps(&result_xyt.x, result);
    //return result_xyt;

    //float d[4]; float b[4];
    //_mm_storeu_ps(d, u_vec);           // d = [dx, dy, dt, 0]
    //_mm_storeu_ps(b, one_minus_u_vec); // b = [bx, by, bt, 0]
    //float dx = d[0], dy = d[1], dt = d[2];
    //float bx = b[0], by = b[1], bt = b[2];
    //float* v = f;

    //float
    //    E = bx * (by * (v[0b000] * bt + v[0b001] * dt) + dy * (v[0b010] * bt + v[0b011] * dt))
    //    + dx * (by * (v[0b100] * bt + v[0b101] * dt) + dy * (v[0b110] * bt + v[0b111] * dt)),
    //    Gx = (v[0b100] - v[0b000]) * by * bt + (v[0b101] - v[0b001]) * by * dt + (v[0b110] - v[0b010]) * dy * bt + (v[0b111] - v[0b011]) * dy * dt,
    //    Gy = (v[0b010] - v[0b000]) * bx * bt + (v[0b011] - v[0b001]) * bx * dt + (v[0b110] - v[0b100]) * dx * bt + (v[0b111] - v[0b101]) * dx * dt,
    //    Gt = (v[0b001] - v[0b000]) * bx * by + (v[0b011] - v[0b010]) * bx * dy + (v[0b101] - v[0b100]) * dx * by + (v[0b111] - v[0b110]) * dx * dy;
    //return { Gx, Gy, Gt, E };
}