#pragma once

#include <vector>
using namespace std;

struct DelaunayPair {
    int id1, id2;
};

/* Iterating over a Delaunay diagram:

    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    ...
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            ... // <- iteration over particles
            indices++;
            id1++;
        }
        int id2 = edges[j];
        ...  // <- iteration over edges
    }

*/
struct DelaunayIterator
{
    int* indices;
    int* edges;
    int num_sites;
    int num_edges;
    int count;
    int id1;

    DelaunayIterator(int num_sites, int num_edges, int* indices, int* edges) {
        this->indices = indices;
        this->edges = edges;
        this->num_sites = num_sites;
        this->num_edges = num_edges;
        this->count = 0;
        this->id1 = 0;
    }

    bool going() const {
        return count < num_edges;
    }

    DelaunayPair next() {
        while (count == indices[id1] && id1 < num_sites) {
            id1++;  // half-iteration over sites is forbidden
        }
        int id2 = edges[count++];
        return { id1, id2 };
    }

    int nextSite(vector<int>& id2_vec) {
        id2_vec.clear();
        while (going()) {
            if (count == indices[id1] || id1 == num_sites) {
                id1++;
                return id1 - 1;
            }
            int id2 = edges[count++];
            id2_vec.push_back(id2);
        }
    }
};


struct SymmetricDelaunay {
    vector<int> indices, edges;

    DelaunayIterator iter() {
        return DelaunayIterator(indices.size(), edges.size(), indices.data(), edges.data());
    }
};

inline SymmetricDelaunay convertAdjacencyVectors(int num_rods, int num_edges, int* offsets, int* higher_neighbors) {
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