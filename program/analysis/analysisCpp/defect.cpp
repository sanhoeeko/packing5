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
    float dx = n_angles / 0.5f;

    memset(output, 0, n_angles * sizeof(int));
    DelaunayIterator it(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);
    while (it.going()) {
        DelaunayPair pair = it.next();
        if ((z_number[pair.id1] == 5 && z_number[pair.id2] == 7) || 
            (z_number[pair.id1] == 7 && z_number[pair.id2] == 5))
        {
            float angle = angle_between_lines(q[pair.id1].t, q[pair.id2].t);  // in [0, 1/2] (mapped from [0, pi/2])
            int interval_index = (int)(angle * dx);
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

/*
    accept: alpha in [-1, 1] from [-pi, pi]
    return: beta in [-1/2, 1/2] from [-pi/2, pi/2] | NAN if alpha = 1/2 (pi/2) or -1/2 (-pi/2)
    equivalent formula: beta = (alpha + 1/2) mod 1 - 1/2
*/
inline static float regular_rotation_angle(float alpha) {
    // validity check: alpha cannot be -1/2 or 1/2 (-pi/2 or pi/2)
    bool valid = abs(abs(alpha) - 0.5f) > 1e-4;
    if (!valid) return NAN;
    if (alpha > 0.5f) {
        return alpha - 1;
    }
    else if (alpha < -0.5f) {
        return alpha + 1;
    }
    else {
        return alpha;
    }
}

/*
     - Core Algorithm - 
    Return: winding number ( = winding angle / (pi/2))
    If one of the rotating angle is pi/2 or -pi/2, return INT_MAX
*/
static int winding_number_2(float X_center, float Y_center, const vector<int>& id2_vec, xyt3f* q, float* angle) {
    AnglePair angle_pairs[16];
    int z = 0;
    for (int id2 : id2_vec) {
        float positional_angle = atan2f(q[id2].y - Y_center, q[id2].x - X_center);
        angle_pairs[z++] = { positional_angle, angle[id2] };
    }
    std::sort(angle_pairs, angle_pairs + z);  // positional angle from low to high

    float prev_arg = modpi01(angle_pairs[z - 1].orientational_angle);
    float total_angle = 0;
    bool valid = true;
    for (int i = 0; i < z; i++) {
        float current_arg = modpi01(angle_pairs[i].orientational_angle);
        float beta = regular_rotation_angle(current_arg - prev_arg);
        if (isnan(beta)) {
            valid = false; break;
        }
        total_angle += beta;
        prev_arg = current_arg;
    }
    if (valid) {
        return round(total_angle);
    }
    else {
        return INT_MAX;
    }
}

void windingNumber2(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_ptr)
{
    // convert Delaunay edge format
    SymmetricDelaunay de = convertAdjacencyVectors(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);
    DelaunayIterator it = de.iter();

    xyt3f* q = (xyt3f*)configuration_ptr;       // length: num_rods
    float* angle = (float*)angle_ptr;           // length: num_rods
    int* output = (int*)output_ptr;             // length: num_rods

    vector<AnglePair> angle_pairs;
    for (int id1 = 0; id1 < it.num_sites; id1++) {
        vector<int> id2_vec = it.neighbors(id1);
        output[id1] = winding_number_2(q[id1].x, q[id1].y, id2_vec, q, angle);
    }
}

struct WindingTriangleResult {
    vector<Triangle> positive, negative, invalid;
};

WindingTriangleResult windingTriangle(TriangleIterator& it, xyt3f* q, float* angle) {
    WindingTriangleResult result;
    while (it.going()) {
        Triangle tri = it.next();
        if (tri.id1 == -1) return result;
        float X_center = (q[tri.id1].x + q[tri.id2].x + q[tri.id3].x) / 3;
        float Y_center = (q[tri.id1].y + q[tri.id2].y + q[tri.id3].y) / 3;
        // s can be 1, -1, 0 or INT_MAX (invalid)
        int s = winding_number_2(X_center, Y_center, { tri.id1, tri.id2, tri.id3 }, q, angle);
        switch (s) {
        case 1:result.positive.push_back(tri); break;
        case -1:result.negative.push_back(tri); break;
        case INT_MAX:result.invalid.push_back(tri); break;
        default: break;
        }
    }
    return result;
}

/*
    return: count of positive defect | count of negative defect
*/
__int64 LCDefectPositions(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_positive_ptr, void* output_negative_ptr) 
{
    xyt3f* q = (xyt3f*)configuration_ptr;                       // length: num_rods
    float* output_positive = (float*)output_positive_ptr;       // length: unknown
    float* output_negative = (float*)output_negative_ptr;       // length: unknown

    TriangleIterator it(num_rods, num_edges, (int*)indices_ptr, (int*)edges_ptr);
    cout << endl;
    WindingTriangleResult result = windingTriangle(it, q, (float*)angle_ptr);

    float* pos_ptr = output_positive;
    for(Triangle& tri : result.positive) {
        float X_center = (q[tri.id1].x + q[tri.id2].x + q[tri.id3].x) / 3;
        float Y_center = (q[tri.id1].y + q[tri.id2].y + q[tri.id3].y) / 3;
        *pos_ptr++ = X_center;
        *pos_ptr++ = Y_center;
    }

    float* neg_ptr = output_negative;
    for (Triangle& tri : result.negative) {
        float X_center = (q[tri.id1].x + q[tri.id2].x + q[tri.id3].x) / 3;
        float Y_center = (q[tri.id1].y + q[tri.id2].y + q[tri.id3].y) / 3;
        *neg_ptr++ = X_center;
        *neg_ptr++ = Y_center;
    }

    int pos_count = result.positive.size();
    int neg_count = result.negative.size();
    return ((__int64)pos_count << 32) | (__int16)neg_count;
}