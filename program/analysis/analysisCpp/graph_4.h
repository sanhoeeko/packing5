#pragma once

#include "defs.h"
#include <string.h>

// This file inherits packing4

typedef int particleId_t;
const int neighbors = 32;

template <int max_neighbors>
struct Graph
{
    struct Fix_length_array
    {
        particleId_t arr[max_neighbors];
        particleId_t &operator[](const int idx) { return arr[idx]; }
    };
    vector<Fix_length_array> data;
    vector<int> z;

    Graph(int n)
    {
        data.resize(n);
        z.resize(n);
        clear();
    }
    void clear()
    {
        memset(data.data(), -1, max_neighbors * data.size() * sizeof(particleId_t)); // -1 = (bin) 1111 1111
        memset(z.data(), 0, z.size() * sizeof(int));
    }
    void add(int i, int j)
    {
        if (z[i] == max_neighbors)
        {
            cout << "Too many neighbors!" << endl;
            throw 1919810;
        }
        else
        {
            data[i][z[i]] = j;
            z[i]++;
        }
    }
    void add_pair(int i, int j)
    {
        add(i, j);
        add(j, i);
    }
    bool has(int i, int j)
    {
        for (int k = 0; k < z[i]; k++)
        {
            if (data[i][k] == j)
                return true;
        }
        return false;
    }
    void add_pair_if_hasnot(int i, int j)
    {
        if (i != j && !has(i, j))
            add_pair(i, j);
    }
};

void delaunayTriangulate(int num_rods, int disks_per_rod, float gamma, xyt3f *ptr, Graph<neighbors> &output_graph);