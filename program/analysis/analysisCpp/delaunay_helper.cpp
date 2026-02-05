#include "pch.h"
#include "defs.h"
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <cmath>
#include <algorithm>
using namespace std;

int DelaunayModulo(int n, int m, int N, void* indices_in_ptr, void* edges_in_ptr, void* mask_ptr, void* indices_out_ptr, 
    void* edges_out_ptr, void* weights_out_ptr) 
{
    /*
        n: number of points
        m: number of edges * 2
        N: number of rods
        mask[k] = 0 => the k-th edge is a bad edge (mask can be NULL)
        indices_in_ptr: n numbers, indptr[1:]
        edges_in_ptr: m numbers
        return: total number of rod edges
    */
    int* indices_in = (int*)indices_in_ptr;
    int* edges_in = (int*)edges_in_ptr;
    int* mask = (int*)mask_ptr;
    int* indices_out = (int*)indices_out_ptr;
    int* edges_out = (int*)edges_out_ptr;
    int* weights_out = (int*)weights_out_ptr;

    vector<unordered_map<int, int>> graph(N);

    // convert python data structure to vector of dict
    int i = 0;
    int i_rod = 0;
    int k_for_next_i = indices_in[0];
    if (mask != NULL) {
        for (int k = 0; k < m; k++) {
            while (k == k_for_next_i && i < n) {
                ++i;
                i_rod = i % N;
                k_for_next_i = indices_in[i];
            }
            if (mask[k]) {
                int j_rod = edges_in[k] % N;
                if (i_rod < j_rod) {
                    graph[i_rod][j_rod]++;
                }
            }
        }
    }
    else {
        for (int k = 0; k < m; k++) {
            while (k == k_for_next_i && i < n) {
                ++i;
                i_rod = i % N;
                k_for_next_i = indices_in[i];
            }
            if (i >= n)break;
            int j = edges_in[k];
            if (j < n) {
                int j_rod = j % N;
                if (i_rod < j_rod) {
                    graph[i_rod][j_rod]++;
                }
            }
        }
    }

    // convert vector of dict to indices, edges and weights
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        cnt += graph[i].size();
        indices_out[i] = cnt;
        for (pair<int, int> kv : graph[i]) {
            *edges_out++ = kv.first;
            *weights_out++ = kv.second;
        }
    }
    return cnt;
}

static inline float cross(const Point& O, const Point& A, const Point& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

vector<int> convexHull(const vector<Point>& points) {
    int n = points.size();
    if (n < 3) return {};

    // generate indices
    vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    sort(indices.begin(), indices.end(), [&](int i, int j) {
        return points[i].x < points[j].x || (points[i].x == points[j].x && points[i].y < points[j].y);
        });

    vector<int> hull;

    // lower hull
    for (int i : indices) {
        while (hull.size() >= 2 && cross(points[hull[hull.size() - 2]], points[hull.back()], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(i);
    }

    // upper hull
    size_t lowerHullSize = hull.size();
    for (int i = n - 2; i >= 0; --i) {
        while (hull.size() > lowerHullSize && cross(points[hull[hull.size() - 2]], points[hull.back()], points[indices[i]]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(indices[i]);
    }

    // remove repetitve points
    if (!hull.empty()) hull.pop_back();

    return hull;
}

void ConvexHull(void* points_ptr, void* out_mask_ptr, int n_points, int n_rods)
{
    int* out_mask = (int*)out_mask_ptr;
    vector<Point> points((Point*)points_ptr, (Point*)points_ptr + n_points);
    vector<int> hull = convexHull(points);
    for (int i : hull) {
        out_mask[i % n_rods] = 1;
    }
}

struct IjkvData
{
    int i, j, k, valid;
};

pair<int, int> findTheThirdPoint(IjkvData* table, int i, int j) {
    /*
        Require: i < j
        table: [[i, j, k, valid], ...], where i < j < k
        return: (position in table, k)
    */
    IjkvData* ptr = table;
    if (i > j)std::swap(i, j);
    while (ptr->valid == 0 || (ptr->i != i && ptr->j != i) || (ptr->j != j && ptr->k != j))ptr++;
    if (ptr->i == i && ptr->j == j) {
        return { ptr - table,ptr->k };
    }
    else if (ptr->j == i && ptr->k == j) {
        return { ptr - table,ptr->i };
    }
    else {
        return { ptr - table,ptr->j };
    }
}

float compute_min_cos(const Point& a, const Point& b, const Point& c) {
    const Point ab = { b.x - a.x, b.y - a.y };
    const Point ac = { c.x - a.x, c.y - a.y };
    const Point bc = { c.x - b.x, c.y - b.y };

    const float len_ab = std::sqrt(ab.x * ab.x + ab.y * ab.y);
    const float len_ac = std::sqrt(ac.x * ac.x + ac.y * ac.y);
    const float len_bc = std::sqrt(bc.x * bc.x + bc.y * bc.y);
    if (len_ab == 0 || len_ac == 0 || len_bc == 0)return -1;

    float cos_alpha = (ab.x * ac.x + ab.y * ac.y) / (len_ab * len_ac);
    float cos_beta = (-ab.x * bc.x - ab.y * bc.y) / (len_ab * len_bc);
    float cos_gamma = (ac.x * bc.x + ac.y * bc.y) / (len_ac * len_bc);
    return std::min(cos_alpha, std::min(cos_beta, cos_gamma));
}

pair<int, int> make_sorted_pair(int a, int b) {
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

void RemoveBadBoundaryEdges(void* points_ptr, void* convex_hull_ptr, void* table_ptr, void* indices_ptr, void* edges_ptr,
    void* mask_ptr, int convex_hull_length, float cos_threshold) 
{
    /*
        Require: i < j
    */
    Point* points = (Point*)points_ptr;
    pair<int, int>* convex_hull = static_cast<pair<int, int>*>(convex_hull_ptr);
    IjkvData* table = (IjkvData*)table_ptr;
    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    int* mask = (int*)mask_ptr;

    queue<pair<int, int>> Q;
    for (int k = 0; k < convex_hull_length; k++) {
        Q.push(convex_hull[k]);
    }
    while (!Q.empty()) {
        int i, j;
        std::tie(i, j) = Q.front(); Q.pop();
        int pos, k;
        std::tie(pos, k) = findTheThirdPoint(table, i, j);  // i<j<k
        float min_cos_of_triangle = compute_min_cos(points[i], points[j], points[k]);
        if (min_cos_of_triangle < cos_threshold) {
            int mask_start = indices[i];
            int mask_end = indices[i + 1];
            for (int l = mask_start; l < mask_end; l++) {
                if (edges[l] == j) {
                    mask[l] = 0; 
                    table[pos].valid = 0;
                    Q.push(make_sorted_pair(i, k)); Q.push(make_sorted_pair(j, k));  // ensure that i<k, j<k
                    break;
                }
            }
        }
    }
}

// code for plot

static inline float sq(float x) { return x * x; }

static int argmin(const float* arr, int size) {
    int idx = 0;
    float min_val = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] < min_val) {
            min_val = arr[i];
            idx = i;
        }
    }
    return idx;
}

void convertXY(int edge_type, float r, float t1, float t2, void* xyxy_ptr) {
    float* xyxy = (float*)xyxy_ptr;
    float x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
    if (edge_type == 0) {  // head-to-head
        float dx1 = r * cos(t1);
        float dy1 = r * sin(t1);
        float x1l = x1 - dx1, x1r = x1 + dx1;
        float y1l = y1 - dy1, y1r = y1 + dy1;

        float dx2 = r * cos(t2);
        float dy2 = r * sin(t2);
        float x2l = x2 - dx2, x2r = x2 + dx2;
        float y2l = y2 - dy2, y2r = y2 + dy2;

        float distances[4] = {
            sq(x1l - x2l) + sq(y1l - y2l),
            sq(x1l - x2r) + sq(y1l - y2r),
            sq(x1r - x2l) + sq(y1r - y2l),
            sq(x1r - x2r) + sq(y1r - y2r)
        };

        switch (argmin(distances, 4)) {
        case 0: x1 = x1l; y1 = y1l; x2 = x2l; y2 = y2l; break;
        case 1: x1 = x1l; y1 = y1l; x2 = x2r; y2 = y2r; break;
        case 2: x1 = x1r; y1 = y1r; x2 = x2l; y2 = y2l; break;
        case 3: x1 = x1r; y1 = y1r; x2 = x2r; y2 = y2r; break;
        }
    }
    else if (edge_type == 1) {  // head-to-side
        float dx1 = r * cos(t1);
        float dy1 = r * sin(t1);
        float x1l = x1 - dx1, x1r = x1 + dx1;
        float y1l = y1 - dy1, y1r = y1 + dy1;

        float dx2 = r * cos(t2);
        float dy2 = r * sin(t2);
        float x2l = x2 - dx2, x2r = x2 + dx2;
        float y2l = y2 - dy2, y2r = y2 + dy2;

        float distances[4] = {
            sq(x1l - x2) + sq(y1l - y2),
            sq(x1r - x2) + sq(y1r - y2),
            sq(x1 - x2l) + sq(y1 - y2l),
            sq(x1 - x2r) + sq(y1 - y2r)
        };

        switch (argmin(distances, 4)) {
        case 0: x1 = x1l; y1 = y1l; break;
        case 1: x1 = x1r; y1 = y1r; break;
        case 2: x2 = x2l; y2 = y2l; break;
        case 3: x2 = x2r; y2 = y2r; break;
        }
    }
    xyxy[0] = x1; xyxy[1] = y1; xyxy[2] = x2; xyxy[3] = y2;
}