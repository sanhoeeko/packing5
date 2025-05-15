#include "pch.h"
#include "defs.h"
#include <math.h>
#include <vector>
#include <algorithm>

inline static float modpi01(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return (y - floor(y));  // return in [0,1)
}

inline static float angle_by_lines(float theta1, float theta2) {
    float delta = modpi01(theta1 - theta2 + pi / 2);
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

struct DelaunayEdges {
    vector<int> indices, edges;
};

DelaunayEdges convertAdjacencyVectors(int num_rods, int num_edges, int* offsets, int* higher_neighbors) {
    vector<vector<int>> adj(num_rods);

    int start = 0, end = offsets[0];
    for (int u = 0; u < num_rods; ++u) {
        for (int j = start; j < end; ++j) {
            int v = higher_neighbors[j];
            adj[u].push_back(v); adj[v].push_back(u);  // add neighbor symmetrically
        }
        start = end;
        end = offsets[u + 1];
    }

    vector<int> new_offsets;
    vector<int> new_neighbors;
    new_offsets.reserve(num_rods);
    new_neighbors.reserve(2 * num_edges);

    for (int u = 0; u < num_rods; ++u) {
        new_neighbors.insert(new_neighbors.end(), adj[u].begin(), adj[u].end());
        new_offsets.push_back(new_neighbors.size());
    }

    return { new_offsets, new_neighbors };
}

inline static float modpi(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return pi * (y - floor(y));
}

struct AnglePair {
    float positional_angle,
        orientational_angle;
    bool operator<(const AnglePair& o) const {
        return positional_angle < o.positional_angle;
    }
    bool operator==(const AnglePair& o) const {
        return positional_angle == o.positional_angle;
    }
};

void sort_and_unique(vector<AnglePair>& vec) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

void windingAngle(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_ptr)
{
    // convert Delaunay edge format
    DelaunayEdges de = convertAdjacencyVectors(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);

    int* indices = de.indices.data();               // length: num_rods
    int* edges = de.edges.data();                   // length: 2 * num_edges
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* angle = (float*)angle_ptr;               // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods

    int id1 = 0;
    vector<AnglePair> angle_pairs;
    for (int j = 0; j < 2 * num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            // calculate the winding angle
            std::sort(angle_pairs.begin(), angle_pairs.end());  // positional angle from low to high
            int z = angle_pairs.size();
            float current_theta = angle_pairs[z - 1].orientational_angle;
            float total_angle = 0;
            for (int i = 0; i < z; i++) {
                float angle_diff = modpi(angle_pairs[i].orientational_angle - current_theta + pi / 2) - pi / 2;
                total_angle += angle_diff;
                current_theta = angle_pairs[i].orientational_angle;
            }
            output[id1] = total_angle;
            // refresh and next
            indices++;
            id1++;
            angle_pairs.clear();
        }
        int id2 = edges[j];
        float positional_angle = atan2f(q[id2].y - q[id1].y, q[id2].x - q[id1].x);
        angle_pairs.push_back({ positional_angle, angle[id2] });
    }
}

void windingAngleNextNearest(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_ptr)
{
    // convert Delaunay edge format
    DelaunayEdges de = convertAdjacencyVectors(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);

    int* indices = de.indices.data();               // length: num_rods
    int* edges = de.edges.data();                   // length: 2 * num_edges
    int* fixed_indices = (int*)indices_ptr;
    xyt3f* q = (xyt3f*)configuration_ptr;           // length: num_rods
    float* angle = (float*)angle_ptr;               // length: num_rods
    float* output = (float*)output_ptr;             // length: num_rods

    int id1 = 0;
    vector<AnglePair> angle_pairs;
    for (int j = 0; j < 2 * num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            // calculate the winding angle
            sort_and_unique(angle_pairs);  // positional angle from low to high
            int z = angle_pairs.size();
            float current_theta = angle_pairs[z - 1].orientational_angle;
            float total_angle = 0;
            for (int i = 0; i < z; i++) {
                float angle_diff = modpi(angle_pairs[i].orientational_angle - current_theta + pi / 2) - pi / 2;
                total_angle += angle_diff;
                current_theta = angle_pairs[i].orientational_angle;
            }
            output[id1] = total_angle;
            // refresh and next
            indices++;
            id1++;
            angle_pairs.clear();
        }
        int id2 = edges[j];
        int id3_start_idx = fixed_indices[id2];
        int id3_end_idx = (id2 == num_rods - 1) ? num_edges : fixed_indices[id2 + 1];
        for (int k = id3_start_idx; k < id3_end_idx; k++) {
            int id3 = edges[k];
            if (id3 == id1)continue;
            float positional_angle = atan2f(q[id3].y - q[id1].y, q[id3].x - q[id1].x);
            angle_pairs.push_back({ positional_angle, angle[id3] });
        }
    }
}