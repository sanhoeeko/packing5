﻿#include "pch.h"
#include "gradient.h"
#include "lbfgs.h"
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

void* addRodShape(int threads, int n, float d, void* p_table, void* p_Vr)
{
    float (*table)[szy][szt] = static_cast<float(*)[szy][szt]>(p_table);
    float* Vr = (float*)p_Vr;
    Rod* rod = new Rod(n, d, table);
    rod->initPotential(threads, Vr);
    return rod;
}

void* addSegmentShape(int threads, float gamma, void* p_table, void* p_Vr)
{
    float (*table)[szy][szt] = static_cast<float(*)[szy][szt]>(p_table);
    float* Vr = (float*)p_Vr;
    Segment* seg = new Segment(gamma, table);
    seg->initPotential(threads, Vr);
    return seg;
}

void delParticleShape(void* particle_shape, int particle_shape_type)
{
    switch (particle_shape_type)
    {
    case RodType: { Rod* rod = (Rod*)particle_shape; delete rod; } break;
    case SegmentType: { Segment* seg = (Segment*)particle_shape; delete seg; } break;
    default:break;
    }
}

void GridLocate(void* p_state, void* p_indices, int x_shift, int y_shift, int cols, int N)
{
    xyt* q = (xyt*)p_state;
    int* indices = (int*)p_indices;
    for (int k = 0; k < N; k++) {
        int i = (int)round(q[k].x / 2) + x_shift;
        int j = (int)round(q[k].y / 2) + y_shift;
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

void CalGradient(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, 
    int lines, int cols, int N)
{
    return calGradient_general<Normal, false>(p_shape, p_state, p_boundary, p_grid, p_Gij, lines, cols, N);
}

void CalGradientAsDisks(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, 
    int lines, int cols, int N)
{
    return calGradient_general<AsDisks, false>(p_shape, p_state, p_boundary, p_grid, p_Gij, lines, cols, N);
}

void CalGradientAndEnergy(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, 
    int lines, int cols, int N)
{
    return calGradient_general<Normal, true>(p_shape, p_state, p_boundary, p_grid, p_Gij, lines, cols, N);
}

typedef L_bfgs<LBFGS_MEMORY_SIZE> LBFGS;

void* CreateLBFGS(int N, void* configuration_src, void* gradient_src)
{
    LBFGS* ptr = new LBFGS(N, (float*)configuration_src, (float*)gradient_src);
    return ptr;
}

void DeleteLBFGS(void* ptr)
{
    LBFGS* lbfgs = (LBFGS*)ptr;
    delete lbfgs;
}

void LbfgsUpdate(void* ptr, void* x_new, void* g_new)
{
    LBFGS* lbfgs = (LBFGS*)ptr;
    lbfgs->update((float*)x_new, (float*)g_new);
}

void LbfgsDirection(void* ptr, void* dst)
{
    LBFGS* lbfgs = (LBFGS*)ptr;
    lbfgs->calDirection_to((float*)dst);
}

float MinDistanceRij(void* p_state, void* p_grid, int lines, int cols, int N)
{
    xyt* state = (xyt*)p_state;
    int* grid = (int*)p_grid;
    return minDistancePP(state, grid, lines, cols, N);
}

float AverageDistanceRij(void* p_state, void* p_grid, int lines, int cols, int N)
{
    xyt* state = (xyt*)p_state;
    int* grid = (int*)p_grid;
    return averageDistancePP(state, grid, lines, cols, N);
}

float MinDistanceRijFull(void* p_state, int N)
{
    return minDistance((xyt*)p_state, N);
}

float RijRatio(void* p_state, void* p_grid, int lines, int cols, int N)
{
    xyt* state = (xyt*)p_state;
    int* grid = (int*)p_grid;
    return distanceRatioPP(state, grid, lines, cols, N);
}

int isOutOfBoundary(void* p_state, void* p_boundary, int N)
{
    xyt* state = (xyt*)p_state;
    EllipticBoundary* boundary = (EllipticBoundary*)p_boundary;
    float a2 = (boundary->a + 1.9) * (boundary->a + 1.9);
    float b2 = (boundary->b + 1.9) * (boundary->b + 1.9);
    for (int i = 0; i < N; i++) {
        float x = state[i].x, y = state[i].y;
        if (x * x / a2 + y * y / b2 > 1) {
            return true;
        }
    }
    return false;
}
