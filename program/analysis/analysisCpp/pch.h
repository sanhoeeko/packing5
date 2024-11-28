#ifndef PCH_H
#define PCH_H

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "framework.h"
#include "voro_interface.h"

DLLEXPORT int disksToVoronoiEdges(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr, 
    float A, float B);
DLLEXPORT int trueDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B);
DLLEXPORT int weightedDelaunay(int num_rods, int disks_per_rod, void* input_points_ptr, void* output_ptr,
    void* output_indices_ptr, float A, float B);
DLLEXPORT void sumOverWeights(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* weights_ptr, void* output_ptr);
DLLEXPORT void sumAntisym(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* weights_ptr, void* output_ptr);
DLLEXPORT void theta_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* configuration_ptr, void* output_ptr);
DLLEXPORT void orientation_diff_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* configuration_ptr, void* output_ptr);


#endif //PCH_H
