#include"pch.h"
#include"defs.h"
#include<math.h>

inline static float modpi(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return (y - floor(y));  // return in [0,1)
}

inline static float angle_by_lines(float theta1, float theta2) {
    float delta = modpi(theta1 - theta2 + pi / 2);
    return abs(delta - 0.5f);  // return in [0,1/2]
}

void Angle57Dist(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    int n_angles, void* xyt_ptr, void* z_number_ptr, void* output_ptr) 
{
    int* indices = (int*)indices_ptr;               // length: num_rods
    int* edges = (int*)edges_ptr;                   // length: num_edges
    int* z_number = (int*)z_number_ptr;             // length: num_edges
    xyt3f* q = (xyt3f*)xyt_ptr;                     // length: num_rods
    int* output = (int*)output_ptr;                 // length: n_angles
    float d_theta = 0.5 / n_angles;

    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        // not only for 5 and 7, but also for 4 and 8, etc
        if ((z_number[id1] < 6 && z_number[id2] > 6) || (z_number[id1] > 6 && z_number[id2] < 6)) {
            float angle = angle_by_lines(q[id1].t, q[id2].t);     // angle in [0, 1/2] (mapped from [0, pi/2])
            int interval_index = angle >= 0.5f ? n_angles - 1 : (int)(angle / d_theta);
            output[interval_index]++;
        }
    }
}