#include "pch.h"
#include "defs.h"
#include "delaunay_iterator.hpp"
#include <math.h>
#include <vector>
#include <algorithm>
#include <string.h>

inline static float modpi01(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return (y - floor(y));  // return in [0,1)
}

inline static float modpi(float x) {
    return modpi01(x) * pi;
}

/*
    Angle between two lines:
        raw angle: alpha = (theta2 - theta1) % pi, where theta1, theta2 are mod pi identical
        if alpha in [0, pi/2] : alpha
        if alpha in [pi/2, pi] : pi - alpha
    Note:
        Swapping theta1, theta2 results in alpha -> pi - alpha, but the interval also changes,
        so the output keeps invariant. The output is thus symmetric about theta1, theta2.
*/
inline static float angle_between_lines(float theta1, float theta2) {
    float alpha = modpi01(theta1 - theta2);  // in [0, 1)
    return alpha < 0.5f ? alpha : 1 - alpha;
}

/*
    Return: histogram in [0, pi), with `n_angles` bins
*/
void Angle57Hist(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    int n_angles, void* xyt_ptr, void* z_number_ptr, void* output_ptr) 
{
    int* z_number = (int*)z_number_ptr;             // length: num_edges
    xyt3f* q = (xyt3f*)xyt_ptr;                     // length: num_rods
    int* output = (int*)output_ptr;                 // length: n_angles
    float d_theta = 0.5 / n_angles;

    memset(output, 0, n_angles * sizeof(int));
    DelaunayIterator it(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);
    while (it.going()) {
        DelaunayPair pair = it.next();
        if ((z_number[pair.id1] == 5 && z_number[pair.id2] == 7) || 
            (z_number[pair.id1] == 7 && z_number[pair.id2] == 5))
        {
            float angle = angle_between_lines(q[pair.id1].t, q[pair.id2].t);  // in [0, 1/2] (mapped from [0, pi/2])
            int interval_index = (int)(angle / d_theta);
            output[interval_index]++;
        }
    }
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

/*
    Return: winding number ( = winding angle / (pi/2))
    If one of the rotating angle is pi/2 or -pi/2, return INT_MAX
*/
void windingNumber2(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_ptr)
{
    // convert Delaunay edge format
    SymmetricDelaunay de = convertAdjacencyVectors(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);

    xyt3f* q = (xyt3f*)configuration_ptr;       // length: num_rods
    float* angle = (float*)angle_ptr;           // length: num_rods
    int* output = (int*)output_ptr;             // length: num_rods

    vector<AnglePair> angle_pairs;
    vector<int> id2_vec;
    DelaunayIterator it = de.iter();
    while (it.going()) {
        int id1 = it.nextSite(id2_vec);
        angle_pairs.clear();
        for (int id2 : id2_vec) {
            float positional_angle = atan2f(q[id2].y - q[id1].y, q[id2].x - q[id1].x);
            angle_pairs.push_back({ positional_angle, angle[id2] });
        }
        std::sort(angle_pairs.begin(), angle_pairs.end());  // positional angle from low to high

        int z = angle_pairs.size();
        float prev_arg = modpi01(angle_pairs[z - 1].orientational_angle);
        float total_angle = 0;
        bool valid = true;
        for (int i = 0; i < z; i++) {
            float current_arg = modpi01(angle_pairs[i].orientational_angle);
            float alpha = current_arg - prev_arg;  // in [-1, 1] -> mapped from [-pi, pi]

            // validity check: alpha cannot be -1/2 or 1/2 (-pi/2 or pi/2)
            valid = abs(abs(alpha) - 0.5f) > 1e-4;  
            if (!valid) break;

            int quadrant = (int)((alpha + 1) * 2);  // in {0, 1, 2, 3}, where 0 => III, 1 => IV, 2 => I, 3 => II
            float beta;  // in [0, 1/2] -> mapped from [0, pi/2]
            switch (quadrant) {
            case 2: beta = alpha; break;  // I quadrant
            case 3: beta = alpha - 1; break;  // II quadrant
            case 0: beta = alpha + 1; break;  // III quadrant
            case 1: beta = alpha; break;  // IV quadrant
            default: throw runtime_error("Invalid quadrant encountered when calculating winding number!");
            }
            total_angle += beta;
            prev_arg = current_arg;
        }
        if (valid) {
            output[id1] = round(total_angle);
        }
        else {
            output[id1] = INT_MAX;
        }
    }
}