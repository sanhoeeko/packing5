#include "pch.h"
#include "gradient.h"
#include <algorithm>
#include <time.h>

void init()
{
    srand(time(0));
    omp_set_nested(1);
}

void* addEllipticBoundary(float a, float b)
{
    EllipticBoundary* boundary = new EllipticBoundary(a, b);
    return boundary;
}

void delEllipticBoundary(void* boundary) {
    EllipticBoundary* _boundary = (EllipticBoundary*)boundary;
    delete _boundary;
}

void setEllipticBoundary(void* boundary, float a, float b) {
    EllipticBoundary* _boundary = (EllipticBoundary*)boundary;
    _boundary->setBoundary(a, b);
}

void* addParticleShape(int threads, int n, float d, void* p_table, void* p_Vr)
{
    float (*table)[szy][szt] = static_cast<float(*)[szy][szt]>(p_table);
    float* Vr = (float*)p_Vr;
    Rod* rod = new Rod(n, d, table);
    rod->initPotential(threads, Vr);
    return rod;
}

void delParticleShape(void* particle_shape)
{
    Rod* rod = (Rod*)particle_shape;
    delete rod;
}

void GridLocate(void* p_state, void* p_indices, int x_shift, int y_shift, int cols, int N)
{
    xyt* q = (xyt*)p_state;
    int* indices = (int*)p_indices;
    for (int k = 0; k < N; k++) {
        int i = (int)round(q[k].x / 2.0f) + x_shift;
        int j = (int)round(q[k].y / 2.0f) + y_shift;
        indices[k] = j * cols + i;
    }
}

void GridTransform(void* p_indices, void* p_grid, int N)
{
    int* indices = (int*)p_indices;
    int* grid = (int*)p_grid;
    for (int i = 0; i < N; i++) {
        int line_start_idx = indices[i] * max_neighbors;
        grid[line_start_idx + grid[line_start_idx] + 1] = i;
        grid[line_start_idx]++;
    }
}

void CalGradient(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
    int lines, int cols, int N)
{
    return calGradient_general<Normal, false>(p_shape, p_state, p_boundary, p_grid, p_Gij, p_z, lines, cols, N);
}

void CalGradientAsDisks(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
    int lines, int cols, int N)
{
    return calGradient_general<AsDisks, false>(p_shape, p_state, p_boundary, p_grid, p_Gij, p_z, lines, cols, N);
}

void StochasticCalGradient(float p, void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
    int lines, int cols, int N)
{
    return stochastic_calGradient_general<Normal>(p_shape, p_state, p_boundary, p_grid, p_Gij, p_z, lines, cols, N, p);
}

void StochasticCalGradientAsDisks(float p, void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z,
    int lines, int cols, int N)
{
    return stochastic_calGradient_general<AsDisks>(p_shape, p_state, p_boundary, p_grid, p_Gij, p_z, lines, cols, N, p);
}

void CalGradientAndEnergy(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
    int lines, int cols, int N)
{
    return calGradient_general<Normal, true>(p_shape, p_state, p_boundary, p_grid, p_Gij, p_z, lines, cols, N);
}
