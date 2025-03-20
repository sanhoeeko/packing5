#include "pch.h"
#include "defs.h"
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
using namespace std;

int DelaunayModulo(int n, int m, int N, void* indices_in_ptr, void* edges_in_ptr, void* mask_ptr, void* indices_out_ptr, 
    void* edges_out_ptr, void* weights_out_ptr) 
{
    /*
        n: number of points
        m: number of edges * 2
        N: number of rods
        mask[k] = 0 => the k-th edge is in the convex hull
        return: total number of edges
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
    return std::min({ cos_alpha, cos_beta, cos_gamma });
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