#include "pch.h"
#include "defs.h"
#include "segdist.h"
#include <vector>
#include <algorithm>
#include <numeric>

template<typename T>
std::vector<size_t> argsort(const std::vector<T>& vec) {
    std::vector<size_t> indices(vec.size()); 
    std::iota(indices.begin(), indices.end(), 0);  // fill in 0, 1, 2, ..., N-1
    std::sort(indices.begin(), indices.end(),
        [&vec](size_t i, size_t j) { return vec[i] < vec[j]; });
    return indices;
}

float point_dist(xyt3f& p, xyt3f q) {
    float dx = p.x - q.x;
    float dy = p.y - q.y;
    return sqrtf(dx * dx + dy * dy);
}

float segment_dist(xyt3f& p, xyt3f& q, float r) {
    return SegDist(r, p.x, p.y, p.t, q.x, q.y, q.t);
}

void correlation(void* xyt_ptr, void* opA_field_ptr, void* opB_field_ptr, void* out_r_ptr, void* out_corr_ptr, int if_seg_dist,
    int N, float gamma, float mean_A, float mean_B, float std_A, float std_B) 
{
    xyt3f* q = (xyt3f*)xyt_ptr;
    float* A_field = (float*)opA_field_ptr;
    float* B_field = (float*)opB_field_ptr;
    float* out_r = (float*)out_r_ptr;
    float* out_corr = (float*)out_corr_ptr;

    float r = 1 - 1 / gamma;
    float coef = 0.5 / (mean_A * mean_B);
    // float coef = 0.5 / (std_A * std_B);
    vector<float> rs; rs.reserve(N * (N - 1) / 2);
    vector<float> corr; corr.reserve(N * (N - 1) / 2);

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dist = if_seg_dist? segment_dist(q[i], q[j], r) : point_dist(q[i], q[j]);
            rs.push_back(dist);
            corr.push_back(coef * ((A_field[i] - mean_A) * (B_field[j] - mean_B) 
                + (A_field[j] - mean_A) * (B_field[i] - mean_B)));
        }
    }
    vector<size_t> indices = argsort(rs);
    for (int i = 0; i < rs.size(); i++) {
        out_r[i] = rs[indices[i]];
        out_corr[i] = corr[indices[i]];
    }
}