#ifndef PCH_H
#define PCH_H

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "framework.h"
#include "defs.h"

DLLEXPORT void init();

DLLEXPORT void* addEllipticBoundary(float a, float b);

DLLEXPORT void delEllipticBoundary(void* boundary);

DLLEXPORT void setEllipticBoundary(void* boundary, float a, float b);

DLLEXPORT void* addParticleShape(int threads, int n, float d, void* p_table, void* p_Vr);

DLLEXPORT void delParticleShape(void* particle_shape);

DLLEXPORT void GridLocate(void* p_state, void* p_indices, int x_shift, int y_shift, int cols, int N);

DLLEXPORT void GridTransform(void* p_indices, void* p_grid, int N);

DLLEXPORT void CalGradient(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, void* p_z, int lines, int cols, int N);

DLLEXPORT void CalGradientAsDisks(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, void* p_z, int lines, int cols, int N);

DLLEXPORT void StochasticCalGradient(float p, void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, void* p_z, int lines, int cols, int N);

DLLEXPORT void StochasticCalGradientAsDisks(float p, void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, void* p_z, int lines, int cols, int N);

DLLEXPORT void CalGradientAndEnergy(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, void* p_z, int lines, int cols, int N);

DLLEXPORT void SumTensor4(void* p_z, void* p_Gij, void* p_gi, int N);

DLLEXPORT void AddVector4(void* p_x, void* p_g, int N, float s);

DLLEXPORT void PerturbVector4(void* p_input, int N, float sigma);

DLLEXPORT void FastClear(void* p_float, int size);

DLLEXPORT void HollowClear(void* p_float, int N, int stride);

DLLEXPORT float FastNorm(void* p_x, int n);

//// test functions /////////////////////////////////////////////////////////////////////////////

DLLEXPORT void preciseGE(void* p_shape, void* scalar_potential, void* scalar_potential_dr, void* p_out,
    float x, float y, float t1, float t2);

DLLEXPORT void interpolateGE(void* p_shape, void* p_out, float x, float y, float t1, float t2);

#endif
