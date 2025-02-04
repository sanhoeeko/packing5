#include "pch.h"
#include "gradient.h"

float minDistance(xyt* q, int N) {
	float current_r2 = 114514;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			float
				dx = q[i].x - q[j].x,
				dy = q[i].y - q[j].y,
				r2 = dx * dx + dy * dy;
			current_r2 = r2 < current_r2 ? r2 : current_r2;
		}
	}
	return sqrt(current_r2);
}

float minDistancePP(xyt* particles, int* grid, int lines, int cols, int N) {
	/*
		Given that a particle is in a certain grid,
		it is only possible to collide with particles in that grid or surrounding grids.
		Remember: index = j * cols + i
	*/
	float current_min_r2 = 114514;

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
							if (r2 < current_min_r2) {
								current_min_r2 = r2;
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
					if (r2 < current_min_r2) {
						current_min_r2 = r2;
					}
				}
			}
		}
	}
	return sqrt(current_min_r2);
}

float averageDistancePP(xyt* particles, int* grid, int lines, int cols, int N) {
	/*
		Given that a particle is in a certain grid,
		it is only possible to collide with particles in that grid or surrounding grids.
		Remember: index = j * cols + i
	*/
	int num = 0;
	float current_r_sum = 0;

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
								current_r_sum += sqrtf(r2); num++;
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
						current_r_sum += sqrtf(r2); num++;
					}
				}
			}
		}
	}
	return current_r_sum / num;
}

float distanceRatioPP(xyt* particles, int* grid, int lines, int cols, int N) {
	/*
		Given that a particle is in a certain grid,
		it is only possible to collide with particles in that grid or surrounding grids.
		Remember: index = j * cols + i
	*/
	int num = 0;
	float current_r_sum = 0;
	float current_min_r = 114514;

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
								float r = sqrtf(r2);
								current_r_sum += r; num++;
								current_min_r = r < current_min_r ? r : current_min_r;
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
						current_r_sum += sqrtf(r2); num++;
					}
				}
			}
		}
	}
	return current_min_r / (current_r_sum / (float)num);
}