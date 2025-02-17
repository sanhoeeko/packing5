#include "pch.h"
#include "segdist.h"
#include "potential.h"

float cross(const v2 a, const v2 b) {
    return a.x * b.y - a.y * b.x;
}

SegmentDist::SegmentDist(float length) : l(length) {
    mat << 1, 1, 1, -1,
        1, -1, -1, 1,
        1, 1, 1, 1,
        1, -1, -1, -1;
}

float SegmentDist::helper(float abac, float bcba, const v2 ab, const v2 ac)
{
    if (abac > 0 && bcba > 0) {
        return std::abs(cross(ac, ab)) / (2 * l);
    }
    else {
        if (abac < 0) {
            return ac.norm();  // AC
        }
        else { // if bcba < 0
            return (ac - ab).norm();  // BC
        }
    }
}

float SegmentDist::inner(float x, float y, float t1, float t2)
{
    float c1 = fcos(t1), c2 = fcos(t2), s1 = fsin(t1), s2 = fsin(t2);
    float c12 = fcos(t1 - t2);
    v2 u1 = { l * c1, l * s1 };
    v2 u2 = { l * c2, l * s2 };
    v2 r = { x, y };

    Vector4f v1(2 * l * l, 2 * l * x * c1, 2 * l * y * s1, 2 * l * l * c12);
    Vector4f v2(2 * l * l, 2 * l * x * c2, 2 * l * y * s2, 2 * l * l * c12);

    Vector4f result1 = mat * v1;
    Vector4f result2 = mat * v2;

    float abac = result1[0], bcba = result1[1], abad = result1[2], bdba = result1[3];
    float dcdb = result2[0], cbcd = result2[1], dcda = result2[2], cacd = result2[3];

    float c_ab = helper(abac, bcba, u1 * 2, r - u2 + u1);
    float d_ab = helper(abad, bdba, u1 * 2, r + u2 + u1);
    float a_cd = helper(cacd, dcda, u2 * 2, r - u2 + u1);
    float b_cd = helper(cbcd, dcdb, u2 * 2, r - u2 - u1);

    return std::min({ c_ab, d_ab, a_cd, b_cd });
}

float SegmentDist::operator()(ParticlePair& p)
{
    return inner(p.x, p.y, p.t1, p.t2);
}
