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
        (i,j,k), (i ¡À 1,j,k), (i,j ¡À 1,k), (i,j,k ¡À 1)
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
        (i,j,k), (i ¡À 1,j,k), (i,j ¡À 1,k), (i,j,k ¡À 1)
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
        fetch potential values of 8 points:
        (i ¡À 1, j ¡À 1, k ¡À 1)
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
        fetch potential values of 8 points:
        (i ¡À 1, j ¡À 1, k ¡À 1)
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
