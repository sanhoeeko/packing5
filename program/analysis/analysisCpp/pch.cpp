#include "pch.h"

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

struct xyt { float x, y, t; };

void z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr, 
    void* output_complex_ptr, float p)
{
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt* q = (xyt*)configuration_ptr;               // length: num_rods
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
    xyt* q = (xyt*)configuration_ptr;               // length: num_rods
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

void anisotropic_z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* orientation_ptr, void* output_complex_ptr, float gamma, float p)
{
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt* q = (xyt*)configuration_ptr;               // length: num_rods
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
            dy1 = gamma * (y * cosf(t1) - x * sinf(t1)),
            dx2 = -x * cosf(t2) - y * sinf(t2),
            dy2 = gamma * (-y * cosf(t2) + x * sinf(t2));
        float
            p_theta_ij = p * atan2f(dy1, dx1),
            p_theta_ji = p * atan2f(dy2, dx2);
        output[4 * j] = cosf(p_theta_ij);
        output[4 * j + 1] = sinf(p_theta_ij);
        output[4 * j + 2] = cosf(p_theta_ji);
        output[4 * j + 3] = sinf(p_theta_ji);
    }
}

void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* complex_ptr, void* output_ptr)
{
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