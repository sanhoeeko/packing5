#include "pch.h"
#include "delaunator.hpp"
#include "graph_4.h"
#include <cassert>

// This file inherits packing4

void _delaunay(int n_sites, vector<float>& points, Graph<neighbors>& graph) 
{
	delaunator::Delaunator d(points);
	for (size_t i = 0; i < d.triangles.size(); i += 3) {
		int p1 = d.triangles[i] % n_sites;
		int p2 = d.triangles[i + 1] % n_sites;
		int p3 = d.triangles[i + 2] % n_sites;
		graph.add_pair_if_hasnot(p1, p2);
		graph.add_pair_if_hasnot(p1, p3);
		graph.add_pair_if_hasnot(p2, p3);
	}
}

void delaunayTriangulate(int num_rods, int disks_per_rod, float gamma, xyt3f* ptr, Graph<neighbors>& graph)
{
	/*
		We use n - point approximation to describe the shape of the rod.
		When the density is higher, more points will be required.
	*/
    assert(disks_per_rod % 2 == 1);		// n must be odd
	int m = (disks_per_rod - 1) / 2;
	float r = 1 - 1 / gamma;
	vector<Point> points; points.resize(num_rods * disks_per_rod);

	// convert rod-like particles to sites
	for (int i = 0; i < num_rods; i++) {
		float
			dx = r * cos(ptr[i].t) / m,
			dy = r * sin(ptr[i].t) / m;
		for (int k = -m; k <= m; k++) {
			float
				x = ptr[i].x + k * dx,
				y = ptr[i].y + k * dy;
			points[(m + k) * num_rods + i] = { x,y };
		}
	}

	// execute the delaunay triangulation
	vector<float> pts = vector<float>((float*)points.data(), (float*)(points.data() + points.size()));
	_delaunay(num_rods, pts, graph);
}
