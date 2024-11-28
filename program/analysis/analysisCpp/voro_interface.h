#pragma once

#include<vector>
#include <utility>
using namespace std;

struct VoronoiEdge {
    int id1, id2;
    float x1, y1, x2, y2;
};

float length(const VoronoiEdge& edge);

typedef vector<pair<int, float>> DelaunayUnit;

vector<VoronoiEdge> PointsToVoronoiEdges(int num_points, float* input_points, float A, float B);
vector<VoronoiEdge> EdgeModulo(const vector<VoronoiEdge>& edges, int num_rods);
vector<DelaunayUnit> TrueDelaunayModulo(const vector<VoronoiEdge>& edges, int num_rods);
vector<DelaunayUnit> WeightedDelaunayModulo(const vector<VoronoiEdge>& edges, int num_rods);