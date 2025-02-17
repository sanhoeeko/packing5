#pragma once

struct v2 { 
    float x, y; 
    v2 operator+(const v2& other) const { return { x + other.x, y + other.y }; }
    v2 operator-(const v2& other) const { return { x - other.x, y - other.y }; }
    v2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
    float norm() const { return std::sqrt(x * x + y * y); }
};

struct SegmentDist {
    Matrix4f mat;
    float l;

    SegmentDist(float length);
    float helper(float abac, float bcba, const v2 ab, const v2 ac);
    float inner(float x, float y, float t1, float t2);
    float operator()(ParticlePair& p);
};