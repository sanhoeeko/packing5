#include "pch.h"
#include "defs.h"
#include <math.h>
#include <string.h>
#include "segdist.h"

/* Iterating over a Delaunay diagram:

    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    ...
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            ... // <- iteration over particles
            indices++;
            id1++;
        }
        int id2 = edges[j];
        ...  // <- iteration over edges
    }

*/

void neighbors(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* output_ptr)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    int* output = (int*)output_ptr;                 // length: num_edges * 2
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[id1]++;
        output[id2]++;
    }
}

void z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, float p)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* output = (float*)output_complex_ptr;     // length: num_edges * 2
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float dx = q[id2].x - q[id1].x;
        float dy = q[id2].y - q[id1].y;
        float p_theta_ij = p * atan2f(dy, dx);
        output[2 * j] = cosf(p_theta_ij);
        output[2 * j + 1] = sinf(p_theta_ij);
    }
}

void orientation_diff_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr, void* output_ptr) {
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* output = (float*)output_ptr;             // length: num_edges
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[j] = q[id2].t - q[id1].t;
    }
}

void symmetricSum(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* a_ptr, void* output_ptr)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    float* a = (float*)a_ptr;                       // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[id1] += a[id2];
        output[id2] += a[id1];
    }
}

void pure_rotation_direction_phi(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_ptr)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods
    int id1 = 0;
    float sum_denominator = 0, sum_numerator = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            output[id1] = 0.5f * atan2f(2 * sum_numerator, sum_denominator);
            sum_denominator = 0; sum_numerator = 0;
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float dx = q[id2].x - q[id1].x;
        float dy = q[id2].y - q[id1].y;
        sum_numerator += dx * dy;
        sum_denominator += dx * dx - dy * dy;
    }
}

void anisotropic_z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* orientation_ptr, void* output_complex_ptr, float gamma, float p)
{
    /*
        return: to be accepted by `sumAnisotropicComplex` in pch.cpp
    */
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* t = (float*)orientation_ptr;             // length: num_rods
    float* output = (float*)output_complex_ptr;     // length: num_edges * 4
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float
            x = q[id2].x - q[id1].x,
            y = q[id2].y - q[id1].y,
            t1 = t[id1],
            t2 = t[id2];
        // this coordinate transform is "rotation by -theta"  
        float
            dx1 = x * cosf(t1) + y * sinf(t1),              
            dy1 = gamma * (-x * sinf(t1) + y * cosf(t1)),
            dx2 = -x * cosf(t2) - y * sinf(t2),
            dy2 = gamma * (x * sinf(t2) - y * cosf(t2));
        float
            p_theta_ij = p * atan2f(dy1, dx1),
            p_theta_ji = p * atan2f(dy2, dx2);
        output[4 * j] = cosf(p_theta_ij);
        output[4 * j + 1] = sinf(p_theta_ij);
        output[4 * j + 2] = cosf(p_theta_ji);
        output[4 * j + 3] = sinf(p_theta_ji);
    }
}

float mean_r_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr)
{
    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    xyt3f* q = (xyt3f*)configuration_ptr;
    float total_rij = 0;
    int rij_cnt = 0;

    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float dx = q[id2].x - q[id1].x;
        float dy = q[id2].y - q[id1].y;
        if (sqrtf(dx * dx + dy * dy) <= 2) {
            total_rij += sqrtf(dx * dx + dy * dy);
            rij_cnt++;
        }
    }
    return total_rij / rij_cnt;
}

float segment_dist_moment(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    float gamma, int moment)
    // moment = 1 or 2
{
    float R = 1 - 1 / gamma;
    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    xyt3f* q = (xyt3f*)configuration_ptr;
    float total_rij = 0;

    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float l = SegDist(R, q[id1].x, q[id1].y, q[id1].t, q[id2].x, q[id2].y, q[id2].t);
        total_rij += powf(l, moment);
    }
    return total_rij / num_edges;
}

void FittedEllipticPhi_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, void* out_gamma, void* out_theta, float p) 
    /*
        void* out_gamma, void* out_theta: can be NULL.
    */
{
    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    xyt3f* q = (xyt3f*)configuration_ptr;
    float* output = (float*)output_complex_ptr;     // length: num_edges * 4

    struct Parameters {
        float a, b, c, d, e, x, y, z;
        void add(float X, float Y) {
            float
                dx = X * X,
                dy = X * Y,
                dz = Y * Y;
            a += dx * dx;  // X * X * X * X;
            b += dx * dy;  // X * X * X * Y;
            c += dy * dy;  // X * X * Y * Y;
            d += dy * dz;  // X * Y * Y * Y;
            e += dz * dz;  // Y * Y * Y * Y;
            x += dx;
            y += dy;
            z += dz;
        }
    };
    Parameters* param = new Parameters[num_rods];
    memset(param, 0, num_rods * sizeof(Parameters));

    struct GammaTheta {
        float gamma, theta;
        void emplace(const Parameters& p) {
            float
                a1 = p.d * p.d - p.c * p.e,
                a2 = p.b * p.e - p.c * p.d,
                a3 = p.c * p.c - p.b * p.d,
                a4 = p.c * p.c - p.a * p.e,
                a5 = p.a * p.d - p.b * p.c,
                a6 = p.b * p.b - p.a * p.c,
                A = a1 * p.x + a2 * p.y + a3 * p.z,
                B = a2 * p.x + a4 * p.y + a5 * p.z,
                C = a3 * p.x + a5 * p.y + a6 * p.z;
            float
                sq_delta = sqrtf(B * B + (A - C) * (A - C)),
                det = a3 * (a3 * a3 - 2 * a2 * a4 - a1 * a5) + a1 * a4 * a4 + a2 * a2 * a5,
                l1 = A + C + sq_delta,
                l2 = A + C - sq_delta;
            if (l1 / det > 0 && l2 / det > 0) {
                float L1 = abs(l1), L2 = abs(l2);
                if (L1 > L2) {
                    gamma = sqrtf(L1 / L2);
                    theta = atan2f(B, A - C - sq_delta);
                }
                else {
                    gamma = sqrtf(L2 / L1);
                    theta = atan2f(B, A - C + sq_delta);
                }
            }
            else {
                gamma = -1;
                theta = 0;
            }
        }
    };
    GammaTheta* gt = new GammaTheta[num_rods];

    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float x = q[id2].x - q[id1].x, y = q[id2].y - q[id1].y;
        param[id1].add(x, y);
        param[id2].add(-x, -y);
    }
    for (int i = 0; i < num_rods; i++) {
        gt[i].emplace(param[i]);
    }
    if (out_gamma != NULL) {
        float* out_g = (float*)out_gamma;
        for (int i = 0; i < num_rods; i++) {
            out_g[i] = gt[i].gamma;
        }
    }
    if (out_theta != NULL) {
        float* out_t = (float*)out_theta;
        for (int i = 0; i < num_rods; i++) {
            out_t[i] = gt[i].theta;
        }
    }
    // calculate anisotropic z_ij
    id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        float x = q[id2].x - q[id1].x, y = q[id2].y - q[id1].y;
        float t1 = gt[id1].theta, t2 = gt[id2].theta;
        // float t1 = q[id1].t, t2 = q[id2].t;
        // this coordinate transform is "rotation by -theta"  
        float
            dx1 = x * cosf(t1) + y * sinf(t1),
            dy1 = gt[id1].gamma * (-x * sinf(t1) + y * cosf(t1)),
            dx2 = -x * cosf(t2) - y * sinf(t2),
            dy2 = gt[id2].gamma * (x * sinf(t2) - y * cosf(t2));
        float
            p_theta_ij = p * atan2f(dy1, dx1),
            p_theta_ji = p * atan2f(dy2, dx2);
        output[4 * j] = cosf(p_theta_ij);
        output[4 * j + 1] = sinf(p_theta_ij);
        output[4 * j + 2] = cosf(p_theta_ji);
        output[4 * j + 3] = sinf(p_theta_ji);
    }
    delete[] param;
    delete[] gt;
}