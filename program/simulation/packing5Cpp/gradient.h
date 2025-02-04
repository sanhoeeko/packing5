#pragma once

#include "gradient_impl.h"

float minDistancePP(xyt* particles, int* grid, int lines, int cols, int N);
float averageDistancePP(xyt* particles, int* grid, int lines, int cols, int N);
float distanceRatioPP(xyt* particles, int* grid, int lines, int cols, int N);
float minDistance(xyt* q, int N);

template<HowToCalGradient how, bool need_energy>
void calGradient_general(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
	int lines, int cols, int N)
{
	Rod* shape = (Rod*)p_shape;
	xyt* q = (xyt*)p_state;
	EllipticBoundary* boundary = (EllipticBoundary*)p_boundary;
	int* grid = (int*)p_grid;
	ge* Gij = (ge*)p_Gij;
	int* z = (int*)p_z;
	collisionDetectPP<how, need_energy>(shape, q, grid, Gij, z, lines, cols, N);
	collisionDetectPW<how, need_energy>(shape, q, boundary, Gij, lines, cols, N);
}

template<HowToCalGradient how>
void stochastic_calGradient_general(void* p_shape, void* p_state, void* p_boundary, void* p_grid, void* p_Gij, void* p_z, 
	int lines, int cols, int N, float p)
{
	Rod* shape = (Rod*)p_shape;
	xyt* q = (xyt*)p_state;
	EllipticBoundary* boundary = (EllipticBoundary*)p_boundary;
	int* grid = (int*)p_grid;
	ge* Gij = (ge*)p_Gij;
	int* z = (int*)p_z;
	stochasticCollisionDetectPP<how>(shape, q, grid, Gij, z, lines, cols, N, p);
	collisionDetectPW<how, false>(shape, q, boundary, Gij, lines, cols, N);
}