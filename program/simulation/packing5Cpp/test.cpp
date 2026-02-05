#include "pch.h"
#include "potential.h"
#include "boundary.h"
#include "segdist.h"

void preciseGE(void* p_shape, void* scalar_potential, void* scalar_potential_dr, void* p_out, 
    float x, float y, float t1, float t2)
{
    Rod* shape = (Rod*)p_shape;
    XytPair* out = (XytPair*)p_out;
    xyt q = {
        x * fcos(t1) + y * fsin(t1),
        -x * fsin(t1) + y * fcos(t1),
        t2 - t1
    };
    *out = shape->StandardGradient(x, y, t1, t2, (float*)scalar_potential_dr);
    out->first.unused = shape->StandardPotential(q, (float*)scalar_potential);
}

void interpolateGE(void* p_shape, void* p_out, float x, float y, float t1, float t2)
{
    Rod* shape = (Rod*)p_shape;
    XytPair* out = (XytPair*)p_out;
    *out = singleGE<Normal, true>(shape, x, y, t1, t2);
}

float standardSegmentDist(void* p_shape, float x, float y, float t1, float t2)
{
    Rod* shape = (Rod*)p_shape;
    float gamma = shape->a / shape->b;
    float r = 1 - 1 / gamma;
    return SegDist(r, 0, 0, t1, x, y, t2);
}
