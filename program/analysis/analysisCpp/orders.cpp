#include "pch.h"
#include <math.h>

struct xyt3f { float x, y, t; };

/* Iterating over a Delaunay diagram:

    int* indices = (int*)indices_ptr + 1;
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

void z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, float p)
{
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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

void sumOverNeighbors(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* a_ptr, void* output_ptr)
{
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods
    int id1 = 0;
    float sum_denominator = 0, sum_numerator = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            output[id1] = 0.5f * atan2f(2 * sum_numerator, sum_denominator);
            indices++;
            id1++;
            sum_denominator = 0; sum_numerator = 0;
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
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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
    int* indices = (int*)indices_ptr + 1;
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