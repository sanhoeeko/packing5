#include "pch.h"
#include "array.h"

vector<float> arange(float start, float stop, float step) {
    int n = int((stop - start) / step);
    n = n > 0 ? n : 0;
    vector<float> res; res.reserve(n);
    for (int i = 0; i < n; i++) {
        res.push_back(start + i * step);
    }
    return res;
}

vector<float> linspace(float start, float stop, int size) {
    float step = (stop - start) / size;
    return arange(start, stop, step);
}

vector<float> linspace_including_endpoint(float start, float stop, int size) {
    float step = (stop - start) / (size - 1);
    return arange(start, stop + step, step);
}

void xyt::operator+=(const xyt& q) {
    x += q.x; y += q.y; t += q.t;
}
void xyt::operator-=(const xyt& q) {
    x -= q.x; y -= q.y; t -= q.t;
}

xyt xyt::operator*(const float a){
    return { a * x,a * y,a * t,a * unused };
}

bool isnan(const xyt& q) {
    return isnan(q.x) || isnan(q.y) || isnan(q.t);
}

bool isinf(const xyt& q) {
    return isinf(q.x) || isinf(q.y) || isinf(q.t);
}