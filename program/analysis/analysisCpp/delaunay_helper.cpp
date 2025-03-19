#include "pch.h"
#include <vector>
#include <utility> // for std::swap


void ConvertToCompressedEdges(int n, int m, void* indices_in_ptr, void* edges_in_ptr, void* indices_out_ptr, void* edges_out_ptr) {
    /*
        n: number of points
        m: number of edges * 2
    */
    int* indices_in = (int*)indices_in_ptr;
    int* edges_in = (int*)edges_in_ptr;
    int* indices_out = (int*)indices_out_ptr;
    int* edges_out = (int*)edges_out_ptr;

    int i = 0;
    int k_for_next_i = indices_in[0];
    int cnt = 0;
    for (int k = 0; k < m; i++) {
        while (k == k_for_next_i && i < n) {
            ++i;
            k_for_next_i = indices_in[i];
            *indices_out++ = cnt;
        }
        if (edges_in[k] > i) {
            *edges_out++ = edges_in[k];
            cnt++;
        }
    }
}

