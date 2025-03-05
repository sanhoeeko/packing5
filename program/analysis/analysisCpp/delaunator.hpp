/*
    [delaunator-cpp]
    This file is downloaded from https://github.com/delfrrr/delaunator-cpp/blob/master/include/delaunator.hpp
*/

#pragma once

#include "defs.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace delaunator {

    typedef float real;  // added by me

//@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
inline size_t fast_mod(const size_t i, const size_t c) {
    return i >= c ? i % c : i;
}

// Kahan and Babuska summation, Neumaier variant; accumulates less FP error
inline real sum(const std::vector<real>& x) {
    real sum = x[0];
    real err = 0.0;

    for (size_t i = 1; i < x.size(); i++) {
        const real k = x[i];
        const real m = sum + k;
        err += std::fabs(sum) >= std::fabs(k) ? sum - m + k : k - m + sum;
        sum = m;
    }
    return sum + err;
}

inline real dist(
    const real ax,
    const real ay,
    const real bx,
    const real by) {
    const real dx = ax - bx;
    const real dy = ay - by;
    return dx * dx + dy * dy;
}

inline real circumradius(
    const real ax,
    const real ay,
    const real bx,
    const real by,
    const real cx,
    const real cy) {
    const real dx = bx - ax;
    const real dy = by - ay;
    const real ex = cx - ax;
    const real ey = cy - ay;

    const real bl = dx * dx + dy * dy;
    const real cl = ex * ex + ey * ey;
    const real d = dx * ey - dy * ex;

    const real x = (ey * bl - dy * cl) * 0.5 / d;
    const real y = (dx * cl - ex * bl) * 0.5 / d;

    if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) && (d > 0.0 || d < 0.0)) {
        return x * x + y * y;
    } else {
        return std::numeric_limits<real>::max();
    }
}

inline bool orient(
    const real px,
    const real py,
    const real qx,
    const real qy,
    const real rx,
    const real ry) {
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
}

inline std::pair<real, real> circumcenter(
    const real ax,
    const real ay,
    const real bx,
    const real by,
    const real cx,
    const real cy) {
    const real dx = bx - ax;
    const real dy = by - ay;
    const real ex = cx - ax;
    const real ey = cy - ay;

    const real bl = dx * dx + dy * dy;
    const real cl = ex * ex + ey * ey;
    const real d = dx * ey - dy * ex;

    const real x = ax + (ey * bl - dy * cl) * 0.5 / d;
    const real y = ay + (dx * cl - ex * bl) * 0.5 / d;

    return std::make_pair(x, y);
}

struct compare {

    std::vector<real> const& coords;
    real cx;
    real cy;

    bool operator()(std::size_t i, std::size_t j) {
        const real d1 = dist(coords[2 * i], coords[2 * i + 1], cx, cy);
        const real d2 = dist(coords[2 * j], coords[2 * j + 1], cx, cy);
        const real diff1 = d1 - d2;
        const real diff2 = coords[2 * i] - coords[2 * j];
        const real diff3 = coords[2 * i + 1] - coords[2 * j + 1];

        if (diff1 > 0.0 || diff1 < 0.0) {
            return diff1 < 0;
        } else if (diff2 > 0.0 || diff2 < 0.0) {
            return diff2 < 0;
        } else {
            return diff3 < 0;
        }
    }
};

inline bool in_circle(
    const real ax,
    const real ay,
    const real bx,
    const real by,
    const real cx,
    const real cy,
    const real px,
    const real py) {
    const real dx = ax - px;
    const real dy = ay - py;
    const real ex = bx - px;
    const real ey = by - py;
    const real fx = cx - px;
    const real fy = cy - py;

    const real ap = dx * dx + dy * dy;
    const real bp = ex * ex + ey * ey;
    const real cp = fx * fx + fy * fy;

    return (dx * (ey * cp - bp * fy) -
            dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx)) < 0.0;
}

constexpr real EPSILON = std::numeric_limits<real>::epsilon();
constexpr std::size_t INVALID_INDEX = std::numeric_limits<std::size_t>::max();

inline bool check_pts_equal(real x1, real y1, real x2, real y2) {
    return std::fabs(x1 - x2) <= EPSILON &&
           std::fabs(y1 - y2) <= EPSILON;
}

// monotonically increases with real angle, but doesn't need expensive trigonometry
inline real pseudo_angle(const real dx, const real dy) {
    const real p = dx / (std::abs(dx) + std::abs(dy));
    return (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0; // [0..1)
}

struct DelaunatorPoint {
    std::size_t i;
    real x;
    real y;
    std::size_t t;
    std::size_t prev;
    std::size_t next;
    bool removed;
};

class Delaunator {

public:
    std::vector<real> const& coords;
    std::vector<std::size_t> triangles;
    std::vector<std::size_t> halfedges;
    std::vector<std::size_t> hull_prev;
    std::vector<std::size_t> hull_next;
    std::vector<std::size_t> hull_tri;
    std::size_t hull_start;

    Delaunator(std::vector<real> const& in_coords);

    real get_hull_area();

private:
    std::vector<std::size_t> m_hash;
    real m_center_x;
    real m_center_y;
    std::size_t m_hash_size;
    std::vector<std::size_t> m_edge_stack;

    std::size_t legalize(std::size_t a);
    std::size_t hash_key(real x, real y) const;
    std::size_t add_triangle(
        std::size_t i0,
        std::size_t i1,
        std::size_t i2,
        std::size_t a,
        std::size_t b,
        std::size_t c);
    void link(std::size_t a, std::size_t b);
};

Delaunator::Delaunator(std::vector<real> const& in_coords)
    : coords(in_coords),
      triangles(),
      halfedges(),
      hull_prev(),
      hull_next(),
      hull_tri(),
      hull_start(),
      m_hash(),
      m_center_x(),
      m_center_y(),
      m_hash_size(),
      m_edge_stack() {
    std::size_t n = coords.size() >> 1;

    real max_x = std::numeric_limits<real>::min();
    real max_y = std::numeric_limits<real>::min();
    real min_x = std::numeric_limits<real>::max();
    real min_y = std::numeric_limits<real>::max();
    std::vector<std::size_t> ids;
    ids.reserve(n);

    for (std::size_t i = 0; i < n; i++) {
        const real x = coords[2 * i];
        const real y = coords[2 * i + 1];

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;

        ids.push_back(i);
    }
    const real cx = (min_x + max_x) / 2;
    const real cy = (min_y + max_y) / 2;
    real min_dist = std::numeric_limits<real>::max();

    std::size_t i0 = INVALID_INDEX;
    std::size_t i1 = INVALID_INDEX;
    std::size_t i2 = INVALID_INDEX;

    // pick a seed point close to the centroid
    for (std::size_t i = 0; i < n; i++) {
        const real d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist) {
            i0 = i;
            min_dist = d;
        }
    }

    const real i0x = coords[2 * i0];
    const real i0y = coords[2 * i0 + 1];

    min_dist = std::numeric_limits<real>::max();

    // find the point closest to the seed
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0) continue;
        const real d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist && d > 0.0) {
            i1 = i;
            min_dist = d;
        }
    }

    real i1x = coords[2 * i1];
    real i1y = coords[2 * i1 + 1];

    real min_radius = std::numeric_limits<real>::max();

    // find the third point which forms the smallest circumcircle with the first two
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0 || i == i1) continue;

        const real r = circumradius(
            i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);

        if (r < min_radius) {
            i2 = i;
            min_radius = r;
        }
    }

    if (!(min_radius < std::numeric_limits<real>::max())) {
        throw std::runtime_error("not triangulation");
    }

    real i2x = coords[2 * i2];
    real i2y = coords[2 * i2 + 1];

    if (orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
        std::swap(i1, i2);
        std::swap(i1x, i2x);
        std::swap(i1y, i2y);
    }

    std::tie(m_center_x, m_center_y) = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);

    // sort the points by distance from the seed triangle circumcenter
    std::sort(ids.begin(), ids.end(), compare{ coords, m_center_x, m_center_y });

    // initialize a hash table for storing edges of the advancing convex hull
    m_hash_size = static_cast<std::size_t>(std::llround(std::ceil(std::sqrt(n))));
    m_hash.resize(m_hash_size);
    std::fill(m_hash.begin(), m_hash.end(), INVALID_INDEX);

    // initialize arrays for tracking the edges of the advancing convex hull
    hull_prev.resize(n);
    hull_next.resize(n);
    hull_tri.resize(n);

    hull_start = i0;

    size_t hull_size = 3;

    hull_next[i0] = hull_prev[i2] = i1;
    hull_next[i1] = hull_prev[i0] = i2;
    hull_next[i2] = hull_prev[i1] = i0;

    hull_tri[i0] = 0;
    hull_tri[i1] = 1;
    hull_tri[i2] = 2;

    m_hash[hash_key(i0x, i0y)] = i0;
    m_hash[hash_key(i1x, i1y)] = i1;
    m_hash[hash_key(i2x, i2y)] = i2;

    std::size_t max_triangles = n < 3 ? 1 : 2 * n - 5;
    triangles.reserve(max_triangles * 3);
    halfedges.reserve(max_triangles * 3);
    add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
    real xp = std::numeric_limits<real>::quiet_NaN();
    real yp = std::numeric_limits<real>::quiet_NaN();
    for (std::size_t k = 0; k < n; k++) {
        const std::size_t i = ids[k];
        const real x = coords[2 * i];
        const real y = coords[2 * i + 1];

        // skip near-duplicate points
        if (k > 0 && check_pts_equal(x, y, xp, yp)) continue;
        xp = x;
        yp = y;

        // skip seed triangle points
        if (
            check_pts_equal(x, y, i0x, i0y) ||
            check_pts_equal(x, y, i1x, i1y) ||
            check_pts_equal(x, y, i2x, i2y)) continue;

        // find a visible edge on the convex hull using edge hash
        std::size_t start = 0;

        size_t key = hash_key(x, y);
        for (size_t j = 0; j < m_hash_size; j++) {
            start = m_hash[fast_mod(key + j, m_hash_size)];
            if (start != INVALID_INDEX && start != hull_next[start]) break;
        }

        start = hull_prev[start];
        size_t e = start;
        size_t q;

        while (q = hull_next[e], !orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1])) { //TODO: does it works in a same way as in JS
            e = q;
            if (e == start) {
                e = INVALID_INDEX;
                break;
            }
        }

        if (e == INVALID_INDEX) continue; // likely a near-duplicate point; skip it

        // add the first triangle from the point
        std::size_t t = add_triangle(
            e,
            i,
            hull_next[e],
            INVALID_INDEX,
            INVALID_INDEX,
            hull_tri[e]);

        hull_tri[i] = legalize(t + 2);
        hull_tri[e] = t;
        hull_size++;

        // walk forward through the hull, adding more triangles and flipping recursively
        std::size_t next = hull_next[e];
        while (
            q = hull_next[next],
            orient(x, y, coords[2 * next], coords[2 * next + 1], coords[2 * q], coords[2 * q + 1])) {
            t = add_triangle(next, i, q, hull_tri[i], INVALID_INDEX, hull_tri[next]);
            hull_tri[i] = legalize(t + 2);
            hull_next[next] = next; // mark as removed
            hull_size--;
            next = q;
        }

        // walk backward from the other side, adding more triangles and flipping
        if (e == start) {
            while (
                q = hull_prev[e],
                orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])) {
                t = add_triangle(q, i, e, INVALID_INDEX, hull_tri[e], hull_tri[q]);
                legalize(t + 2);
                hull_tri[q] = t;
                hull_next[e] = e; // mark as removed
                hull_size--;
                e = q;
            }
        }

        // update the hull indices
        hull_prev[i] = e;
        hull_start = e;
        hull_prev[next] = i;
        hull_next[e] = i;
        hull_next[i] = next;

        m_hash[hash_key(x, y)] = i;
        m_hash[hash_key(coords[2 * e], coords[2 * e + 1])] = e;
    }
}

real Delaunator::get_hull_area() {
    std::vector<real> hull_area;
    size_t e = hull_start;
    do {
        hull_area.push_back((coords[2 * e] - coords[2 * hull_prev[e]]) * (coords[2 * e + 1] + coords[2 * hull_prev[e] + 1]));
        e = hull_next[e];
    } while (e != hull_start);
    return sum(hull_area);
}

std::size_t Delaunator::legalize(std::size_t a) {
    std::size_t i = 0;
    std::size_t ar = 0;
    m_edge_stack.clear();

    // recursion eliminated with a fixed-size stack
    while (true) {
        const size_t b = halfedges[a];

        /* if the pair of triangles doesn't satisfy the Delaunay condition
        * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
        * then do the same check/flip recursively for the new pair of triangles
        *
        *           pl                    pl
        *          /||\                  /  \
        *       al/ || \bl            al/    \a
        *        /  ||  \              /      \
        *       /  a||b  \    flip    /___ar___\
        *     p0\   ||   /p1   =>   p0\---bl---/p1
        *        \  ||  /              \      /
        *       ar\ || /br             b\    /br
        *          \||/                  \  /
        *           pr                    pr
        */
        const size_t a0 = 3 * (a / 3);
        ar = a0 + (a + 2) % 3;

        if (b == INVALID_INDEX) {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                //i = INVALID_INDEX;
                break;
            }
        }

        const size_t b0 = 3 * (b / 3);
        const size_t al = a0 + (a + 1) % 3;
        const size_t bl = b0 + (b + 2) % 3;

        const std::size_t p0 = triangles[ar];
        const std::size_t pr = triangles[a];
        const std::size_t pl = triangles[al];
        const std::size_t p1 = triangles[bl];

        const bool illegal = in_circle(
            coords[2 * p0],
            coords[2 * p0 + 1],
            coords[2 * pr],
            coords[2 * pr + 1],
            coords[2 * pl],
            coords[2 * pl + 1],
            coords[2 * p1],
            coords[2 * p1 + 1]);

        if (illegal) {
            triangles[a] = p1;
            triangles[b] = p0;

            auto hbl = halfedges[bl];

            // edge swapped on the other side of the hull (rare); fix the halfedge reference
            if (hbl == INVALID_INDEX) {
                std::size_t e = hull_start;
                do {
                    if (hull_tri[e] == bl) {
                        hull_tri[e] = a;
                        break;
                    }
                    e = hull_next[e];
                } while (e != hull_start);
            }
            link(a, hbl);
            link(b, halfedges[ar]);
            link(ar, bl);
            std::size_t br = b0 + (b + 1) % 3;

            if (i < m_edge_stack.size()) {
                m_edge_stack[i] = br;
            } else {
                m_edge_stack.push_back(br);
            }
            i++;

        } else {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                break;
            }
        }
    }
    return ar;
}

inline std::size_t Delaunator::hash_key(const real x, const real y) const {
    const real dx = x - m_center_x;
    const real dy = y - m_center_y;
    return fast_mod(
        static_cast<std::size_t>(std::llround(std::floor(pseudo_angle(dx, dy) * static_cast<real>(m_hash_size)))),
        m_hash_size);
}

std::size_t Delaunator::add_triangle(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t a,
    std::size_t b,
    std::size_t c) {
    std::size_t t = triangles.size();
    triangles.push_back(i0);
    triangles.push_back(i1);
    triangles.push_back(i2);
    link(t, a);
    link(t + 1, b);
    link(t + 2, c);
    return t;
}

void Delaunator::link(const std::size_t a, const std::size_t b) {
    std::size_t s = halfedges.size();
    if (a == s) {
        halfedges.push_back(b);
    } else if (a < s) {
        halfedges[a] = b;
    } else {
        throw std::runtime_error("Cannot link edge");
    }
    if (b != INVALID_INDEX) {
        std::size_t s2 = halfedges.size();
        if (b == s2) {
            halfedges.push_back(a);
        } else if (b < s2) {
            halfedges[b] = a;
        } else {
            throw std::runtime_error("Cannot link edge");
        }
    }
}

} //namespace delaunator
