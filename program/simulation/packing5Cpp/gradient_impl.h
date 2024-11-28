#pragma once

#include "defs.h"
#include "potential.h"
#include "boundary.h"
#include <random>

template<HowToCalGradient how, bool need_energy>
void collisionDetectPP(Rod* shape, xyt* particles, int* grid, ge* Gij, int* z, int lines, int cols, int N) {
	/*
		Given that a particle is in a certain grid,
		it is only possible to collide with particles in that grid or surrounding grids.
		Remember: index = j * cols + i
	*/
	int global_start = cols;
	int global_end = (lines - 1) * cols;
	int collision_detect_region[4] = { -cols + 1,1,cols + 1,cols };

	for (int i = global_start; i < global_end; i++)
	{
		int len_cell = grid[i * max_neighbors];
		if (len_cell != 0)
		{
			int* cell = 1 + grid + i * max_neighbors;
			// for "half-each" surrounding grid (not 9, but 5 cells, including itself): 4 cells different from self
			for (int k = 0; k < 4; k++)
			{
				int j = i + collision_detect_region[k];
				int len_next_cell = grid[j * max_neighbors];
				if (len_next_cell != 0)
				{
					int* next_cell = 1 + grid + j * max_neighbors;
					// for each particle pair in these two grid:
					for (int ii = 0; ii < len_cell; ii++) {
						for (int jj = 0; jj < len_next_cell; jj++) {
							int
								p = cell[ii], q = next_cell[jj];
							xyt*
								P = particles + p, * Q = particles + q;
							float
								dx = P->x - Q->x,
								dy = P->y - Q->y,
								r2 = dx * dx + dy * dy;
							if (r2 < 4) {
								XytPair g = singleGE<how, need_energy>(shape, dx, dy, P->t, Q->t);
								int pm = p * max_neighbors, qm = q * max_neighbors;
								Gij[pm + z[p]++] = g.first;
								Gij[qm + z[q]++] = g.second;
							}
						}
					}
				}
			}
			// When and only when collide in one cell, triangular loop must be taken,
			// which ensure that no collision is calculated twice.
			for (int ii = 0; ii < len_cell; ii++) {
				for (int jj = ii + 1; jj < len_cell; jj++) {
					int
						p = cell[ii], q = cell[jj];
					xyt*
						P = particles + p, * Q = particles + q;
					float
						dx = P->x - Q->x,
						dy = P->y - Q->y,
						r2 = dx * dx + dy * dy;
					if (r2 < 4) {
						XytPair g = singleGE<how, need_energy>(shape, dx, dy, P->t, Q->t);
						int pm = p * max_neighbors, qm = q * max_neighbors;
						Gij[pm + z[p]++] = g.first;
						Gij[qm + z[q]++] = g.second;
					}
				}
			}
		}
	}
}

template<HowToCalGradient how, bool need_energy>
void collisionDetectPW(Rod* shape, xyt* particles, EllipticBoundary* b, ge* Gij, int lines, int cols, int N) {
	for (int p = 0; p < N; p++) {
		if (b->maybeCollide(particles[p])) {
			Gij[(p + 1) * max_neighbors - 1] = b->collide<how, need_energy>(shape, particles[p]);
		}
	}
}

template<HowToCalGradient how>
void stochasticCollisionDetectPP(Rod* shape, xyt* particles, int* grid, ge* Gij, int* z, int lines, int cols, int N, float p) {
	int global_start = cols;
	int global_end = (lines - 1) * cols;
	int threshold = (int)round(p * RAND_MAX);
	int collision_detect_region[4] = { -cols + 1,1,cols + 1,cols };
	std::default_random_engine rand;

	for (int i = global_start; i < global_end; i++)
	{
		int len_cell = grid[i * max_neighbors];
		if (len_cell != 0 && rand() < threshold)
		{
			int* cell = 1 + grid + i * max_neighbors;
			// for "half-each" surrounding grid (not 9, but 5 cells, including itself): 4 cells different from self
			for (int k = 0; k < 4; k++)
			{
				int j = i + collision_detect_region[k];
				int len_next_cell = grid[j * max_neighbors];
				if (len_next_cell != 0)
				{
					int* next_cell = 1 + grid + j * max_neighbors;
					// for each particle pair in these two grid:
					for (int ii = 0; ii < len_cell; ii++) {
						for (int jj = 0; jj < len_next_cell; jj++) {
							int
								p = cell[ii], q = next_cell[jj];
							xyt*
								P = particles + p, * Q = particles + q;
							float
								dx = P->x - Q->x,
								dy = P->y - Q->y,
								r2 = dx * dx + dy * dy;
							if (r2 < 4) {
								XytPair g = singleGE<how, false>(shape, dx, dy, P->t, Q->t);
								int pm = p * max_neighbors, qm = q * max_neighbors;
								Gij[pm + z[p]++] = g.first;
								Gij[qm + z[q]++] = g.second;
							}
						}
					}
				}
			}
			// When and only when collide in one cell, triangular loop must be taken,
			// which ensure that no collision is calculated twice.
			for (int ii = 0; ii < len_cell; ii++) {
				for (int jj = ii + 1; jj < len_cell; jj++) {
					int
						p = cell[ii], q = cell[jj];
					xyt*
						P = particles + p, * Q = particles + q;
					float
						dx = P->x - Q->x,
						dy = P->y - Q->y,
						r2 = dx * dx + dy * dy;
					if (r2 < 4) {
						XytPair g = singleGE<how, false>(shape, dx, dy, P->t, Q->t);
						int pm = p * max_neighbors, qm = q * max_neighbors;
						Gij[pm + z[p]++] = g.first;
						Gij[qm + z[q]++] = g.second;
					}
				}
			}
		}
	}
}