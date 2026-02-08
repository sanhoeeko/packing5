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

    vector<int> neighbors(int id1_input) {
        int start = id1_input == 0 ? 0 : indices[id1_input - 1];
        int end = indices[id1_input];
        return vector<int>(edges + start, edges + end);
    }
};

struct Triangle {
    int id1, id2, id3;
};

struct TriangleIterator
{
    int* indices;
    int* edges;
    int num_sites;
    int num_edges;
    int count;
    int id1, n1_start, n1_end;
    int id2, n2_start, n2_end;
    int tri_buffer_size;
    Triangle tri_buffer[2];  // for a given pair of (id1, id2), there are at most 2 possible triangles

    TriangleIterator(int num_sites, int num_edges, int* indices, int* edges) {
        this->indices = indices;
        this->edges = edges;
        this->num_sites = num_sites;
        this->num_edges = num_edges;
        this->count = 0;
        this->id1 = 0;
        this->tri_buffer_size = 0;
    }

    bool going() const {
        return count < num_edges;
    }

    void _fetch_triangles() {
        while (count == indices[id1] && id1 < num_sites) {
            id1++;
            n1_start = _get_starting_idx(id1);
            n1_end = indices[id1];
        }
        id2 = edges[count++];
        n2_start = _get_starting_idx(id2);
        n2_end = indices[id2];
        // find the common neighbor id3: if id3 in id1.neighbors() and id3 in id2.neighbors()
        // the order id1 < id2 < id3 is automatically satisfied if the delaunay is ordered, not symmetric
        tri_buffer_size = 0;
        for (int i = n1_start; i < n1_end; i++) {
            for (int j = n2_start; j < n2_end; j++) {
                if (edges[i] == edges[j]) {
                    int id3 = edges[i];
                    tri_buffer[tri_buffer_size] = { id1,id2,id3 };
                    tri_buffer_size++;
                    if (tri_buffer_size == 2)break;
                }
            }
            if (tri_buffer_size == 2)break;
        }
    }

    Triangle next() {
        while (tri_buffer_size == 0) {
            if (!going()) {
                return { -1, -1, -1 };  // terminal signal
            }
            _fetch_triangles();
        }
        tri_buffer_size--;
        return tri_buffer[tri_buffer_size];
    }

private:
    int _get_starting_idx(int id1_input) {
        return id1_input == 0 ? 0 : indices[id1_input - 1];
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