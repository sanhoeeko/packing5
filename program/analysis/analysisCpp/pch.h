#ifndef PCH_H
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
DLLEXPORT void anisotropic_z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* orientation_ptr, void* output_complex_ptr, float gamma, float p);
DLLEXPORT void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* complex_ptr, void* output_ptr);
DLLEXPORT float RijRatio(void* p_xyt, int N);

#endif //PCH_H
