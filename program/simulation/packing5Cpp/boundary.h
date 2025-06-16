#pragma once

#include "defs.h"
#include "potential.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

struct EllipticBoundary {
    float a, b;
    float a2, b2, inv_inner_a2, inv_inner_b2;
    int if_a_less_than_b;

    EllipticBoundary(float a, float b);
    void setBoundary(float a, float b);
    bool maybeCollide(const xyt& particle);
    float distOutOfBoundary(const xyt& particle);
    void solveNearestPointOnEllipse(float x1, float y1, float& x0, float& y0);
    template<HowToCalGradient how, bool need_energy, bool elliptic> ge collide(Rod* shape, const xyt& q);
};

template<HowToCalGradient how, bool need_energy, bool elliptic>
ge EllipticBoundary::collide(Rod* shape, const xyt& q)
{
	const float h_min = -0.01;
	float x0, y0, absx0, absy0;

	// q.x,	q.y cannot be both zero because of the `maybeCollide` guard. 
	if constexpr (!elliptic) {
		// For circular boundary
		float r = sqrt(q.x * q.x + q.y * q.y);
		if (r < a - 1) return { 0,0,0,0 };
		float ratio = a / r;
		x0 = q.x * ratio;
		y0 = q.y * ratio;
	}
	else {
		// For elliptic boundary
		float absx1 = abs(q.x), absy1 = abs(q.y);
		if (absx1 < 1e-3) {
			x0 = 0; y0 = q.y > 0 ? b : -b;
		}
		else if (absy1 < 1e-3) {
			y0 = 0; x0 = q.x > 0 ? a : -a;
		}
		else {
			solveNearestPointOnEllipse(absx1, absy1, absx0, absy0);
			x0 = q.x > 0 ? absx0 : -absx0;
			y0 = q.y > 0 ? absy0 : -absy0;
		}
	}

	// check if really collide: if not, return nothing
	float
		dx = q.x - x0,
		dy = q.y - y0,
		r2 = dx * dx + dy * dy;
	if (r2 >= 1) {
		return { 0, 0, 0, 0 };
	}

	// calculate the mirror image
	float
		alpha = atan2f(a2 * y0, b2 * x0),	// the angle of the tangent line
		beta = q.t,
		thetap = 2 * alpha - beta;

	// calculate the gradient
	ge g = singleGE<how, need_energy>(shape, 2 * dx, 2 * dy, q.t, thetap).first;

	// check if the particle is outside the boundary. if so, add a penalty
	// This should not be frequently used in real systems
	float h = distOutOfBoundary(q);
	if (h > h_min) {
		float fr = -10 * (expf(h - h_min) - 1);
		g += { fr* q.x, fr* q.y, 0, 0 };
	}
	return g;
}