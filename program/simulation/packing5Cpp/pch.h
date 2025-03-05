#ifndef PCH_H
#define PCH_H

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#include "framework.h"
#else
#define DLLEXPORT extern "C"
#endif

#include "defs.h"

struct ForceTorque { float force, torque; };

DLLEXPORT void init();

DLLEXPORT void* addEllipticBoundary(float a, float b);

DLLEXPORT void delEllipticBoundary(void* boundary);

DLLEXPORT void setEllipticBoundary(void* boundary, float a, float b);

DLLEXPORT void* addRodShape(int threads, int n, float d, void* p_table, void* p_Vr);

DLLEXPORT void* addSegmentShape(int threads, float gamma, void* p_table, void* p_Vr);

DLLEXPORT void delParticleShape(void* particle_shape, int particle_shape_type);

DLLEXPORT void GridLocate(void* p_state, void* p_indices, int x_shift, int y_shift, int cols, int N);

DLLEXPORT void GridTransform(void* p_indices, void* p_grid, int N);

DLLEXPORT void CalGradient(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, int lines, int cols, int N);

DLLEXPORT void CalGradientAsDisks(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, int lines, int cols, int N);

DLLEXPORT void CalGradientAndEnergy(void* p_shape, void* p_state, void* p_boundary, void* p_grid, 
    void* p_Gij, int lines, int cols, int N);

DLLEXPORT float MinDistanceRij(void* p_state, void* p_grid, int lines, int cols, int N);

DLLEXPORT float AverageDistanceRij(void* p_state, void* p_grid, int lines, int cols, int N);

DLLEXPORT float MinDistanceRijFull(void* p_state, int N);

DLLEXPORT float RijRatio(void* p_state, void* p_grid, int lines, int cols, int N);

DLLEXPORT int isOutOfBoundary(void* p_state, void* p_boundary, int N);

DLLEXPORT void ClipGradient(void* q_ptr, int N);

DLLEXPORT void* CreateLBFGS(int N, void* configuration_src, void* gradient_src);

DLLEXPORT void DeleteLBFGS(void* ptr);

DLLEXPORT void LbfgsInit(void* ptr, float initial_stepsize);

DLLEXPORT void LbfgsUpdate(void* ptr);

DLLEXPORT void LbfgsDirection(void* ptr, void* dst);

DLLEXPORT void SumTensor4(void* p_Gij, void* p_gi, int N);

DLLEXPORT void AddVector4(void* p_x, void* p_g, void* p_dst, int N, float s);

DLLEXPORT void AddVector4FT(void* p_x, void* p_g, void* p_dst, int N, float s_force, float s_torque);

DLLEXPORT void PerturbVector4(void* p_input, int N, float sigma);

DLLEXPORT void FastClear(void* p_float, int size);

DLLEXPORT void HollowClear(void* p_float, int N, int stride);

DLLEXPORT float FastNorm(void* p_x, int n);

DLLEXPORT ForceTorque FastMaxFT(void* p_x, int n);

DLLEXPORT void FastMask(void* p_x, void* p_mask, int n);

DLLEXPORT void GenerateMask(void* p_mask, int size, float p);

DLLEXPORT float MaxAbsVector4(void* p_x, int n);

DLLEXPORT void CwiseMulVector4(void* p_g, int N, float s);

DLLEXPORT void AverageState(float temperature, void* p_state, void* energies, void* dst, int N, int n_samples);

DLLEXPORT float AverageStateZeroTemperature(void* p_state, void* energies, void* dst, int N, int n_samples);

//// test functions /////////////////////////////////////////////////////////////////////////////

DLLEXPORT void preciseGE(void* p_shape, void* scalar_potential, void* scalar_potential_dr, void* p_out,
    float x, float y, float t1, float t2);

DLLEXPORT void interpolateGE(void* p_shape, void* p_out, float x, float y, float t1, float t2);

#endif
