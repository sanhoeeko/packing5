#pragma once

#include "defs.h"
#include "potential.h"
#include <cmath>
#include <cstdlib>

struct EllipticBoundary {
    float a, b;
    float a2, b2, inv_inner_a2, inv_inner_b2;
    bool if_a_less_than_b;

    EllipticBoundary(float a, float b);
    void setBoundary(float a, float b);
    bool maybeCollide(const xyt& particle);
    float distOutOfBoundary(const xyt& particle);
    void solveNearestPointOnEllipse(float x1, float y1, float& x0, float& y0);
    template<HowToCalGradient how, bool need_energy> ge collide(Rod* shape, const xyt& q);
};

template<HowToCalGradient how, bool need_energy>
ge EllipticBoundary::collide(Rod* shape, const xyt& q)
{
	float x0, y0, absx0, absy0;

	// check if the particle is outside the boundary. if so, return a penalty
	// a penalty is marked by {id2 = -114514, theta1 = h}
	float h = distOutOfBoundary(q);
	if (h > 0) {
		float fr = -10 * (expf(h) - 1);
		return { fr * q.x, fr * q.y, 0, 0 };
	}

	// q.x,	q.y cannot be both zero because of the `maybeCollide` guard. 
	float absx1 = abs(q.x), absy1 = abs(q.y);
	if (absx1 < 1e-4) {
		x0 = 0; y0 = q.y > 0 ? b : -b;
	}
	else if (absy1 < 1e-4) {
		y0 = 0; x0 = q.x > 0 ? a : -a;
	}
	else {
		solveNearestPointOnEllipse(absx1, absy1, absx0, absy0);
		x0 = q.x > 0 ? absx0 : -absx0;
		y0 = q.y > 0 ? absy0 : -absy0;
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
	return singleGE<how, need_energy>(shape, 2 * dx, 2 * dy, q.t, thetap).first;
}