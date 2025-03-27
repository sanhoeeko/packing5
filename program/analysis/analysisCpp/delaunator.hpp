#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace delaunator {

// Kahan and Babuska summation, Neumaier variant; accumulates less FP error
inline float sum(const std::vector<float>& x) {
    float sum = x[0];
    float err = 0.0;

    for (int i = 1; i < x.size(); i++) {
        const float k = x[i];
        const float m = sum + k;
        err += std::fabs(sum) >= std::fabs(k) ? sum - m + k : k - m + sum;
        sum = m;
    }
    return sum + err;
}

inline float dist(
    const float ax,
    const float ay,
    const float bx,
    const float by) {
    const float dx = ax - bx;
    const float dy = ay - by;
    return dx * dx + dy * dy;
}

inline float circumradius(
    const float ax,
    const float ay,
    const float bx,
    const float by,
    const float cx,
    const float cy) {
    const float dx = bx - ax;
    const float dy = by - ay;
    const float ex = cx - ax;
    const float ey = cy - ay;

    const float bl = dx * dx + dy * dy;
    const float cl = ex * ex + ey * ey;
    const float d = dx * ey - dy * ex;

    const float x = (ey * bl - dy * cl) * 0.5 / d;
    const float y = (dx * cl - ex * bl) * 0.5 / d;

    if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) && (d > 0.0 || d < 0.0)) {
        return x * x + y * y;
    } else {
        return std::numeric_limits<float>::max();
    }
}

inline bool orient(
    const float px,
    const float py,
    const float qx,
    const float qy,
    const float rx,
    const float ry) {
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
}

inline std::pair<float, float> circumcenter(
    const float ax,
    const float ay,
    const float bx,
    const float by,
    const float cx,
    const float cy) {
    const float dx = bx - ax;
    const float dy = by - ay;
    const float ex = cx - ax;
    const float ey = cy - ay;

    const float bl = dx * dx + dy * dy;
    const float cl = ex * ex + ey * ey;
    const float d = dx * ey - dy * ex;

    const float x = ax + (ey * bl - dy * cl) * 0.5 / d;
    const float y = ay + (dx * cl - ex * bl) * 0.5 / d;

    return std::make_pair(x, y);
}

inline float compare(
    std::vector<float> const& coords,
    int i,
    int j,
    float cx,
    float cy) {
    const float d1 = dist(coords[2 * i], coords[2 * i + 1], cx, cy);
    const float d2 = dist(coords[2 * j], coords[2 * j + 1], cx, cy);
    const float diff1 = d1 - d2;
    const float diff2 = coords[2 * i] - coords[2 * j];
    const float diff3 = coords[2 * i + 1] - coords[2 * j + 1];

    if (diff1 > 0.0 || diff1 < 0.0) {
        return diff1;
    } else if (diff2 > 0.0 || diff2 < 0.0) {
        return diff2;
    } else {
        return diff3;
    }
}

struct sort_to_center {

    std::vector<float> const& coords;
    float cx;
    float cy;

    bool operator()(int i, int j) {
        return compare(coords, i, j, cx, cy) < 0;
    }
};

inline bool in_circle(
    float ax,
    float ay,
    float bx,
    float by,
    float cx,
    float cy,
    float px,
    float py) {
    const float dx = ax - px;
    const float dy = ay - py;
    const float ex = bx - px;
    const float ey = by - py;
    const float fx = cx - px;
    const float fy = cy - py;

    const float ap = dx * dx + dy * dy;
    const float bp = ex * ex + ey * ey;
    const float cp = fx * fx + fy * fy;

    return (dx * (ey * cp - bp * fy) -
            dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx)) < 0.0;
}

constexpr float EPSILON = std::numeric_limits<float>::epsilon();
constexpr int INVALID_INDEX = std::numeric_limits<int>::max();

inline bool check_pts_equal(float x1, float y1, float x2, float y2) {
    return std::fabs(x1 - x2) <= EPSILON &&
           std::fabs(y1 - y2) <= EPSILON;
}

// monotonically increases with real angle, but doesn't need expensive trigonometry
inline float pseudo_angle(float dx, float dy) {
    const float p = dx / (std::abs(dx) + std::abs(dy));
    return (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0; // [0..1)
}

struct DelaunatorPoint {
    int i;
    float x;
    float y;
    int t;
    int prev;
    int next;
    bool removed;
};

class Delaunator {

public:
    std::vector<float> const& coords;
    std::vector<int> triangles;
    std::vector<int> halfedges;
    std::vector<int> hull_prev;
    std::vector<int> hull_next;
    std::vector<int> hull_tri;
    int hull_start;

    Delaunator(std::vector<float> const& in_coords);

    float get_hull_area();

private:
    std::vector<int> m_hash;
    float m_center_x;
    float m_center_y;
    int m_hash_size;

    int legalize(int a);
    int hash_key(float x, float y);
    int add_triangle(
        int i0,
        int i1,
        int i2,
        int a,
        int b,
        int c);
    void link(int a, int b);
};

Delaunator::Delaunator(std::vector<float> const& in_coords)
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
      m_hash_size() {
    int n = coords.size() >> 1;

    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    std::vector<int> ids;
    ids.reserve(n);

    for (int i = 0; i < n; i++) {
        const float x = coords[2 * i];
        const float y = coords[2 * i + 1];

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;

        ids.push_back(i);
    }
    const float cx = (min_x + max_x) / 2;
    const float cy = (min_y + max_y) / 2;
    float min_dist = std::numeric_limits<float>::max();

    int i0 = INVALID_INDEX;
    int i1 = INVALID_INDEX;
    int i2 = INVALID_INDEX;

    // pick a seed point close to the centroid
    for (int i = 0; i < n; i++) {
        const float d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist) {
            i0 = i;
            min_dist = d;
        }
    }

    const float i0x = coords[2 * i0];
    const float i0y = coords[2 * i0 + 1];

    min_dist = std::numeric_limits<float>::max();

    // find the point closest to the seed
    for (int i = 0; i < n; i++) {
        if (i == i0) continue;
        const float d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist && d > 0.0) {
            i1 = i;
            min_dist = d;
        }
    }

    float i1x = coords[2 * i1];
    float i1y = coords[2 * i1 + 1];

    float min_radius = std::numeric_limits<float>::max();

    // find the third point which forms the smallest circumcircle with the first two
    for (int i = 0; i < n; i++) {
        if (i == i0 || i == i1) continue;

        const float r = circumradius(
            i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);

        if (r < min_radius) {
            i2 = i;
            min_radius = r;
        }
    }

    if (!(min_radius < std::numeric_limits<float>::max())) {
        throw std::runtime_error("not triangulation");
    }

    float i2x = coords[2 * i2];
    float i2y = coords[2 * i2 + 1];

    if (orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
        std::swap(i1, i2);
        std::swap(i1x, i2x);
        std::swap(i1y, i2y);
    }

    std::tie(m_center_x, m_center_y) = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);

    // sort the points by distance from the seed triangle circumcenter
    std::sort(ids.begin(), ids.end(), sort_to_center{ coords, m_center_x, m_center_y });

    // initialize a hash table for storing edges of the advancing convex hull
    m_hash_size = static_cast<int>(std::llround(std::ceil(std::sqrt(n))));
    m_hash.resize(m_hash_size);
    std::fill(m_hash.begin(), m_hash.end(), INVALID_INDEX);

    // initialize arrays for tracking the edges of the advancing convex hull
    hull_prev.resize(n);
    hull_next.resize(n);
    hull_tri.resize(n);

    hull_start = i0;

    int hull_size = 3;

    hull_next[i0] = hull_prev[i2] = i1;
    hull_next[i1] = hull_prev[i0] = i2;
    hull_next[i2] = hull_prev[i1] = i0;

    hull_tri[i0] = 0;
    hull_tri[i1] = 1;
    hull_tri[i2] = 2;

    m_hash[hash_key(i0x, i0y)] = i0;
    m_hash[hash_key(i1x, i1y)] = i1;
    m_hash[hash_key(i2x, i2y)] = i2;

    int max_triangles = n < 3 ? 1 : 2 * n - 5;
    triangles.reserve(max_triangles * 3);
    halfedges.reserve(max_triangles * 3);
    add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
    float xp = std::numeric_limits<float>::quiet_NaN();
    float yp = std::numeric_limits<float>::quiet_NaN();
    for (int k = 0; k < n; k++) {
        const int i = ids[k];
        const float x = coords[2 * i];
        const float y = coords[2 * i + 1];

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
        int start = 0;

        int key = hash_key(x, y);
        for (int j = 0; j < m_hash_size; j++) {
            start = m_hash[(key + j) % m_hash_size];
            if (start != INVALID_INDEX && start != hull_next[start]) break;
        }

        start = hull_prev[start];
        int e = start;
        int q;

        while (q = hull_next[e], !orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1])) { //TODO: does it works in a same way as in JS
            e = q;
            if (e == start) {
                e = INVALID_INDEX;
                break;
            }
        }

        if (e == INVALID_INDEX) continue; // likely a near-duplicate point; skip it

        // add the first triangle from the point
        int t = add_triangle(
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
        int next = hull_next[e];
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

float Delaunator::get_hull_area() {
    std::vector<float> hull_area;
    int e = hull_start;
    do {
        hull_area.push_back((coords[2 * e] - coords[2 * hull_prev[e]]) * (coords[2 * e + 1] + coords[2 * hull_prev[e] + 1]));
        e = hull_next[e];
    } while (e != hull_start);
    return sum(hull_area);
}

int Delaunator::legalize(int a) {
    const int b = halfedges[a];

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
    const int a0 = a - a % 3;
    const int b0 = b - b % 3;

    const int al = a0 + (a + 1) % 3;
    const int ar = a0 + (a + 2) % 3;
    const int bl = b0 + (b + 2) % 3;

    const int p0 = triangles[ar];
    const int pr = triangles[a];
    const int pl = triangles[al];
    const int p1 = triangles[bl];

    if (b == INVALID_INDEX) {
        return ar;
    }

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
            int e = hull_start;
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

        int br = b0 + (b + 1) % 3;

        legalize(a);
        return legalize(br);
    }
    return ar;
}

int Delaunator::hash_key(float x, float y) {
    const float dx = x - m_center_x;
    const float dy = y - m_center_y;
    return static_cast<int>(std::llround(
               std::floor(pseudo_angle(dx, dy) * static_cast<float>(m_hash_size)))) %
           m_hash_size;
}

int Delaunator::add_triangle(
    int i0,
    int i1,
    int i2,
    int a,
    int b,
    int c) {
    int t = triangles.size();
    triangles.push_back(i0);
    triangles.push_back(i1);
    triangles.push_back(i2);
    link(t, a);
    link(t + 1, b);
    link(t + 2, c);
    return t;
}

void Delaunator::link(int a, int b) {
    int s = halfedges.size();
    if (b != INVALID_INDEX) {  // modified here
        if (a == s) {
            halfedges.push_back(b);
        }
        else if (a < s) {
            halfedges[a] = b;
        }
        /*else {
            throw std::runtime_error("Cannot link edge");
        }*/

        int s2 = halfedges.size();
        if (b == s2) {
            halfedges.push_back(a);
        }
        else if (b < s2) {
            halfedges[b] = a;
        }
        /*else {
            throw std::runtime_error("Cannot link edge");
        }*/
    }
}

} //namespace delaunator
