﻿#include "pch.h"
#include "defs.h"
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

void sumOverWeights(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* weights_ptr, void* output_ptr)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    float* weights = (float*)weights_ptr;           // length: num_edges
    float* output = (float*)output_ptr;             // length: num_rods
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[id1] += weights[j];
        output[id2] += weights[j];
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

void complexSum(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* complex_ptr, void* output_ptr)
{
    /*
        for Phi_{ij} = Phi_{ji}
    */
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    float* cplx = (float*)complex_ptr;              // length: num_edges * 2
    float* output = (float*)output_ptr;             // length: num_rods * 2
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[2 * id1] += cplx[2 * j];
        output[2 * id1 + 1] += cplx[2 * j + 1];
        output[2 * id2] += cplx[2 * j];
        output[2 * id2 + 1] += cplx[2 * j + 1];
    }
}

void symmetricMax(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* a_ptr, void* output_ptr)
{
    // assume that original data [a] satisfies a > 0
    // and [output] has been set to zero in Python
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
        output[id1] = a[j] > output[id1] ? a[j] : output[id1];
        output[id2] = a[j] > output[id2] ? a[j] : output[id2];
    }
}

void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* complex_ptr, void* output_ptr)
{
    /*
        for Phi_{ij} does not equal Phi_{ji}
    */
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    float* cplx = (float*)complex_ptr;              // length: num_edges * 4
    float* output = (float*)output_ptr;             // length: num_rods * 2
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[2 * id1] += cplx[4 * j];
        output[2 * id1 + 1] += cplx[4 * j + 1];
        output[2 * id2] += cplx[4 * j + 2];
        output[2 * id2 + 1] += cplx[4 * j + 3];
    }
}

/*  
    return: r = min_rij / ave_rij. 0 < r <= 1.
*/
float RijRatio(void* p_xyt, int N)
{
    xyt3f* q = (xyt3f*)p_xyt;
    float current_min_rij = 114514;
    float current_total_rij = 0;
    int current_rij_cnt = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            float
                dx = q[i].x - q[j].x,
                dy = q[i].y - q[j].y,
                rij = sqrtf(dx * dx + dy * dy);
            if (rij < 2) {
                current_min_rij = rij < current_min_rij ? rij : current_min_rij;
                current_total_rij += rij;
                current_rij_cnt++;
            }
        }
    }
    return current_min_rij / (current_total_rij / current_rij_cnt);
}

int isOutOfBoundary(void* p_xyt, int N, float A, float B)
{
    xyt3f* q = (xyt3f*)p_xyt;
    float A2 = A * A, B2 = B * B;
    for (int i = 0; i < N; i++) {
        if (q[i].x * q[i].x / A2 + q[i].y * q[i].y / B2 >= 1) return 1;
    }
    return 0;
}

float CubicMinimum(float a, float b, float c, float d) {
    // This code is generated by deepseek.
    const float eps = 1e-6f;

    // cubic
    if (fabs(a) > eps) {
        float discriminant = b * b - 3 * a * c;

        if (discriminant <= eps) return numeric_limits<float>::quiet_NaN();

        float sqrt_part = sqrt(discriminant);
        float x = (-b + sqrt_part) / (3 * a);

        float second_deriv = 6 * a * x + 2 * b;
        return (second_deriv > eps) ? x : numeric_limits<float>::quiet_NaN();
    }
    // quadratic
    else if (fabs(b) > eps) {
        return (b > eps) ? -c / (2 * b) : numeric_limits<float>::quiet_NaN();
    }
    // linear or constant
    else {
        return numeric_limits<float>::quiet_NaN();
    }
}

float distToEllipse(float a, float b, float x1, float y1) {
    /*
        Formulae:
        the point (x0, y0) on the ellipse cloest to (x1, y1) in the first quadrant:

            x0 = a2*x1 / (t+a2)
            y0 = b2*y1 / (t+b2)

        where t is the root of

            ((a*x1)/(t+a2))^2 + ((b*y1)/(t+b2))^2 - 1 = 0

        in the range of t > -b*b. The initial guess can be t0 = -b*b + b*y1.
    */
    // float t_prolate = -b2 + b * y1;
    // float t_oblate = -a2 + a * x1;
    float a2 = a * a, b2 = b * b;
    float t = a < b ? (-a2 + a * x1) : (-b2 + b * y1);

    for (int i = 0; i < 16; i++) {
        // Newton root finding. There is always `Ga * Ga + Gb * Gb - 1 > 0`.
        // There must be MORE iterations for particles near principal axes.
        float
            a2pt = a2 + t,
            b2pt = b2 + t,
            ax1 = a * x1,
            by1 = b * y1,
            Ga = ax1 / a2pt,
            Gb = by1 / b2pt,
            G = Ga * Ga + Gb * Gb - 1,
            dG = -2 * ((ax1 * ax1) / (a2pt * a2pt * a2pt) + (by1 * by1) / (b2pt * b2pt * b2pt));
        if (G < 1e-3f) {
            break;
        }
        else {
            t -= G / dG;
        }
    }
    float
        x0 = a2 * x1 / (t + a2),
        y0 = b2 * y1 / (t + b2),
        dx = x1 - x0,
        dy = y1 - y0;
    return sqrtf(dx * dx + dy * dy);
}

void DistToEllipse(float a, float b, void* points_ptr, void* out_ptr, int N) {
    Point* points = (Point*)points_ptr;
    float* out = (float*)out_ptr;
    for (int i = 0; i < N; i++) {
        out[i] = distToEllipse(a, b, abs(points[i].x), abs(points[i].y));
    }
}

/*
    Require: values are positive integers
*/
void vote(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* values_ptr, void* output_ptr, int max_value)
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    int* values = (int*)values_ptr;                 // length: num_rods
    int* output = (int*)output_ptr;                 // length: num_rods
    int cols = max_value + 1;
    int* count = new int[cols * num_rods]();
    int id1 = 0;
    // count
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        count[id1 * cols + values[id2]]++;
        count[id2 * cols + values[id1]]++;
    }
    // sort (smaller value has higher piority)
    for (int i = 0; i < num_rods; i++) {
        int max_v = -1;
        int max_count = 0;
        for (int v = 0; v <= max_value; v++) {
            if (count[i * cols + v] > max_count) {
                max_count = count[i * cols + v];
                max_v = v;
            }
        }
        output[i] = max_v;
    }
    // clear
    delete[] count;
}