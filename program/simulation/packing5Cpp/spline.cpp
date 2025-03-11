#include "pch.h"
#include "defs.h"
#include "spline.h"
#include "potential.h"

static int confine(int x, int a, int b){
    return x < a ? a : (x > b ? b : x);
}

__m128 cubic_B_spline(float dx, float dy, float dt, const float(&C)[4][4][4]) {
    return _sumGE(dx, dy, dt, C, std::make_integer_sequence<int, 64>{});
}

xyt ParticleShape::interpolatePotentialCBS(const xyt& q)
{
    if (q.y >= 1)return { 0,0,0,0 };
    float C[4][4][4];
    ge Ge;
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    // find the nearest cell: [i,i+3] x [j,j+3] x [k,k+3]
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = confine(floor(X) - 1, 0, szx - 4),
        j = confine(floor(Y) - 1, 0, szy - 4),
        k = confine(floor(T) - 1, 0, szt - 4);
    float
        dx = (X - i) / a1,
        dy = (Y - j) / a2,
        dt = (T - k) / a3;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // some memcpy
        }
    }
    __m128 Ge_vec = cubic_B_spline(dx, dy, dt, C);
    _mm_storeu_ps(&Ge.x, Ge_vec);
    return Ge;
}