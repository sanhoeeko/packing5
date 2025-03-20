#include "pch.h"
#include <vector>
#include <unordered_map>
using namespace std;

#include<iostream>

int DelaunayModulo(int n, int m, int N, void* indices_in_ptr, void* edges_in_ptr, void* hull_ptr, void* indices_out_ptr, 
    void* edges_out_ptr, void* weights_out_ptr) 
{
    /*
        n: number of points
        m: number of edges * 2
        N: number of rods
        hull: provided by `ConvertConvexHull`, can be NULL if we don't care the convex hull
        return: total number of edges
    */
    int* indices_in = (int*)indices_in_ptr;
    int* edges_in = (int*)edges_in_ptr;
    pair<int, int>* hull = static_cast<pair<int, int>*>(hull_ptr);
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
        int j = edges_in[k];
        int j_rod = j % N;
        if (i_rod < j_rod) {
            graph[i_rod][j_rod]++;
        }
    }

    // prune edges that form convex hull
    /*for (int i = 0; i < n; i++) {
        if (hull[i].first != -1) {
            i_rod = i % N;
            int j_rod_1 = hull[i].first % N, j_rod_2 = hull[i].second % N;
            graph[i_rod][j_rod_1]--;
            if (graph[i_rod][j_rod_1] <= 0) graph[i_rod].erase(j_rod_1);
            graph[i_rod][j_rod_2]--;
            if (graph[i_rod][j_rod_2] <= 0) graph[i_rod].erase(j_rod_2);
        }
    }*/

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

void ConvertConvexHull(void* convex_hull_ptr, void* out_ptr, int convex_hull_length, int n_points) {
    /*
        convex_hull: [[id1, id2], [id1, id2], ...]
        out: initialized with -1. output with [[a, b], [-1, -1], ...]
        which means 0 is on the convex hull and connected with a and b, 1 is not on the convex hull.
    */
    pair<int, int>* convex_hull = static_cast<pair<int, int>*>(convex_hull_ptr);
    pair<int, int>* out = static_cast<pair<int, int>*>(out_ptr);
    for (int i = 0; i < convex_hull_length; i++) {
        int id1, id2;
        std::tie(id1, id2) = convex_hull[i];
        if (out[id1].first == -1) {
            out[id1].first = id2;
        }
        else {
            out[id1].second = id2;
        }
        if (out[id2].first == -1) {
            out[id2].first = id1;
        }
        else {
            out[id2].second = id1;
        }
    }
}