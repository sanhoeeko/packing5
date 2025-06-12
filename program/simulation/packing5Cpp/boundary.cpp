#include "pch.h"
#include "boundary.h"

#define _sign(__x) (__x >= 0)

EllipticBoundary::EllipticBoundary(float a, float b)
{
	setBoundary(a, b);
}

void EllipticBoundary::setBoundary(float a, float b)
{
	const float eps = 1e-9;
	this->a = a; this->b = b;

	// derived
	a2 = a * a; b2 = b * b;
	inv_inner_a2 = 1 / ((a - 2) * (a - 2));
	inv_inner_b2 = 1 / ((b - 2) * (b - 2));
	if (abs(a - b) < eps) {
		if_a_less_than_b = 0;
	}
	else {
		if_a_less_than_b = a < b ? 1 : -1;
	}
}

bool EllipticBoundary::maybeCollide(const xyt& q)
{
	return (q.x) * (q.x) * inv_inner_a2 + (q.y) * (q.y) * inv_inner_b2 > 1;
}

float EllipticBoundary::distOutOfBoundary(const xyt& q)
{
	float f = (q.x) * (q.x) / a2 + (q.y) * (q.y) / b2 - 1;
	return f;
}

/*
	require: (x1, y1) in the first quadrant
*/
void EllipticBoundary::solveNearestPointOnEllipse(float x1, float y1, float& x0, float& y0) {
	/*
		Formulae:
		the point (x0, y0) on the ellipse cloest to (x1, y1) in the first quadrant:

			x0 = a2*x1 / (t+a2)
			y0 = b2*y1 / (t+b2)

		where t is the root of

			((a*x1)/(t+a2))^2 + ((b*y1)/(t+b2))^2 - 1 = 0

		in the range of t > -b*b. The initial guess can be t0 = -b*b + b*y1.
	*/
	// float t_prolate = -b2 + b * y1;
	// float t_oblate = -a2 + a * x1;
	float t = _sign(if_a_less_than_b) ? (-a2 + a * x1) : (-b2 + b * y1);

	for (int i = 0; i < 16; i++) {
		// Newton root finding. There is always `Ga * Ga + Gb * Gb - 1 > 0`.
		// There must be MORE iterations for particles near principal axes.
		float
			a2pt = a2 + t,
			b2pt = b2 + t,
			ax1 = a * x1,
			by1 = b * y1,
			Ga = ax1 / a2pt,
			Gb = by1 / b2pt,
			G = Ga * Ga + Gb * Gb - 1,
			dG = -2 * ((ax1 * ax1) / (a2pt * a2pt * a2pt) + (by1 * by1) / (b2pt * b2pt * b2pt));
		if (G < 1e-3f) {
			break;
		}
		else {
			t -= G / dG;
		}
	}
	x0 = a2 * x1 / (t + a2);
	y0 = b2 * y1 / (t + b2);
}