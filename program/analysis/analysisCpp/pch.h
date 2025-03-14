﻿#ifndef PCH_H
#define PCH_H

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#include "framework.h"
#else
#define DLLEXPORT extern "C"
#endif

#include "voro_interface.h"

DLLEXPORT int disksToVoronoiEdges(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr, 
    float A, float B);
DLLEXPORT int trueDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B);
DLLEXPORT int weightedDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B);
DLLEXPORT int legacyDelaunay(int num_rods, int disks_per_rod, float gamma, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr);
DLLEXPORT void sumOverWeights(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* weights_ptr, void* output_ptr);
DLLEXPORT void sumComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* complex_ptr, void* output_ptr);
DLLEXPORT void z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, float p);
DLLEXPORT void orientation_diff_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* configuration_ptr, void* output_ptr);
DLLEXPORT void sumOverNeighbors(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* a_ptr, void* output_ptr);
DLLEXPORT void pure_rotation_direction_phi(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_ptr);
DLLEXPORT void anisotropic_z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* orientation_ptr, void* output_complex_ptr, float gamma, float p);
DLLEXPORT void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* complex_ptr, void* output_ptr);
DLLEXPORT float mean_r_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr);
DLLEXPORT float segment_dist_moment(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    float gamma, int moment);
DLLEXPORT float RijRatio(void* p_xyt, int N);
DLLEXPORT int isOutOfBoundary(void* p_xyt, int N, float A, float B);
DLLEXPORT float CubicMinimum(float a, float b, float c, float d);

#endif //PCH_H
