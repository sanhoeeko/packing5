#pragma once

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

    DelaunayIterator(int* indices, int* edges, int num_sites, int num_edges) {
        this->indices = indices;
        this->edges = edges;
        this->num_sites = num_sites;
        this->num_edges = num_edges;
        this->count = 0;
        this->id1 = 0;
    }

    bool end() const {
        return count == num_edges;
    }

    DelaunayPair next() {
        while (count == indices[id1] && id1 < num_sites) {
            id1++;  // half-iteration over sites is forbidden
        }
        int id2 = edges[count++];
        return { id1, id2 };
    }
};