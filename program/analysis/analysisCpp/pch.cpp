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

void sumAntisym(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* weights_ptr, void* output_ptr)
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
        output[id2] -= weights[j];
    }
}

struct xyt { float x, y, t; };

void theta_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr, void* output_ptr) {
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt* q = (xyt*)configuration_ptr;               // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        output[j] = atan2f(q[id2].y - q[id1].y, q[id2].x - q[id1].x);
    }
}

void orientation_diff_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr, void* output_ptr) {
    int* indices = (int*)indices_ptr + 1;           // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    xyt* q = (xyt*)configuration_ptr;               // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods
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