#include"pch.h"
#include"defs.h"

inline static float modpi_2(float x) {
    const float a = 2 / pi;
    float y = x * a;
    return y - floor(y);
}

void Angle57Dist(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    int n_angles, void* xyt_ptr, void* z_number_ptr, void* output_ptr) 
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    int* z_number = (int*)z_number_ptr;             // length: num_edges
    xyt3f* q = (xyt3f*)xyt_ptr;                     // length: num_rods
    int* output = (int*)output_ptr;                 // length: n_angles
    float d_theta = 1.0 / n_angles;   // Note that modpi_2 returns y in [0,1)

    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        if (z_number[id1] == 5 && z_number[id2] == 7 || z_number[id1] == 7 && z_number[id2] == 5) {
            float angle = modpi_2(q[id1].t - q[id2].t);     // angle in [0, 1) (mapped from [0, pi/2))
            int interval_index = (int)floor(angle / d_theta);
            output[interval_index]++;
        }
    }
}