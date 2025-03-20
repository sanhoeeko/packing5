#include "pch.h"
#include <vector>
#include <unordered_map>
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

