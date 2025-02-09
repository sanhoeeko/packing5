#include "pch.h"
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

struct xyt3f { float x, y, t; };

/*
    return: number of Voronoi ridges
*/
int disksToVoronoiEdges(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr, float A, float B)
{
    float* input_points = (float*)input_points_ptr;
    vector<VoronoiEdge> edges = EdgeModulo(
        PointsToVoronoiEdges(num_rods * disks_per_rod, input_points, A, B), 
        num_rods
    );
    int n_edges = edges.size();
    memcpy(output_ptr, edges.data(), n_edges * sizeof(VoronoiEdge));
    return n_edges;
}

/*
    return: number of Delaunay links
*/
int trueDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B) 
{
    float* input_points = (float*)input_points_ptr;
    int* output_indices = (int*)output_indices_ptr;
    pair<int, float>* output = (pair<int, float>*)output_ptr;
    vector<DelaunayUnit> links = TrueDelaunayModulo(
        PointsToVoronoiEdges(num_rods * disks_per_rod, input_points, A, B), 
        num_rods
    );
    int total_size = 0;
    for (auto& d_unit : links) {
        int d_size = d_unit.size();
        *output_indices++ = total_size;
        memcpy(output + total_size, d_unit.data(), sizeof(pair<int, float>) * d_size);
        total_size += d_size;
    }
    return total_size;
}

/*
    return: number of Delaunay links
*/
int weightedDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B)
{
    float* input_points = (float*)input_points_ptr;
    int* output_indices = (int*)output_indices_ptr;
    pair<int, float>* output = (pair<int, float>*)output_ptr;
    vector<DelaunayUnit> links = WeightedDelaunayModulo(
        PointsToVoronoiEdges(num_rods * disks_per_rod, input_points, A, B),
        num_rods
    );
    int total_size = 0;
    for (auto& d_unit : links) {
        int d_size = d_unit.size();
        *output_indices++ = total_size;
        memcpy(output + total_size, d_unit.data(), sizeof(pair<int, float>) * d_size);
        total_size += d_size;
    }
    return total_size;
}

void sumOverWeights(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* weights_ptr, void* output_ptr)
{
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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

void sumComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* complex_ptr, void* output_ptr)
{
    /*
        for Phi_{ij} = Phi_{ji}
    */
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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

void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* complex_ptr, void* output_ptr)
{
    /*
        for Phi_{ij} does not equal Phi_{ji}
    */
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
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