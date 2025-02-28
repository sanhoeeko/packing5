#include "pch.h"
#include "voro_interface.h"
#include <string.h>
#include <immintrin.h>
#include <unordered_map>

#define JC_VORONOI_IMPLEMENTATION
#include"jc_voronoi.h"

struct Point { float x, y; };

float length(const VoronoiEdge& edge)
{
    float dx = edge.x1 - edge.x2;
    float dy = edge.y1 - edge.y2;
    return sqrtf(dx * dx + dy * dy);
}

template<int ratio>
float precisionUpto(float x)
{
    return round(x * ratio) / ratio;
}

/*
    Not very useful for large datasets.
*/
void perturb_equal_values(int n_points, Point* pts, float epsilon = 1e-3) 
{
    const int prec = 100;      // precision
    std::unordered_map<float, int> value_counts_x;
    std::unordered_map<float, int> value_counts_y;

    // count values
    for (int i = 0; i < n_points; i++) {
        value_counts_x[precisionUpto<prec>(pts[i].x)]++;
        value_counts_y[precisionUpto<prec>(pts[i].y)]++;
    }

    for (int i = 0; i < n_points; i++) {
        float approx_x = precisionUpto<prec>(pts[i].x);
        float approx_y = precisionUpto<prec>(pts[i].y);
        pts[i].x += (value_counts_x[approx_x] - 1) * epsilon;
        value_counts_x[approx_x]--;
        pts[i].y += (value_counts_y[approx_y] - 1) * epsilon;
        value_counts_y[approx_y]--;
    }
}

/*
    Interface of jc_voronoi
    Input_points fomat: x, y, x, y, ...
*/
vector<VoronoiEdge> PointsToVoronoiEdges(int num_points, float* input_points, float A, float B)
{
    // perturb_equal_values(num_points, (Point*)input_points);     // pre-process

    jcv_point* points = (jcv_point*)input_points;
    jcv_rect bounding_box = { { -A, -B }, { A, B } };
    jcv_diagram diagram;
    memset(&diagram, 0, sizeof(jcv_diagram));
    vector<VoronoiEdge> result; result.reserve(8 * num_points);

    jcv_diagram_generate(num_points, points, &bounding_box, NULL, &diagram);
    const jcv_site* sites = jcv_diagram_get_sites(&diagram);

    for (int i = 0; i < diagram.numsites; i++) {
        jcv_graphedge* graph_edge = sites[i].edges;
        while (graph_edge) {
            if (graph_edge->neighbor != NULL 
                && graph_edge->neighbor[1].index >= 0
                && graph_edge->neighbor[1].index < num_points)
            {  
                // NULL neighbor => boundary edges
                // abnormal neighbor[1].index => edges connected with boundary
                // graph_edge->edge->sites : jcv_site_*[2] => input points at both sides of the edge
                VoronoiEdge e = {
                    graph_edge->edge->sites[0]->index,
                    graph_edge->edge->sites[1]->index,
                };
                // copy float 4
                __m128 pos_data = _mm_load_ps((float*)&graph_edge->pos[0]);
                _mm_store_ps(&e.x1, pos_data);
                // save and roll the list
                result.push_back(e);
            }
            graph_edge = graph_edge->next;
        }
    }

    jcv_diagram_free(&diagram);
    return result;
}

vector<VoronoiEdge> EdgeModulo(const vector<VoronoiEdge>& edges, int num_rods)
{
    vector<VoronoiEdge> result; result.reserve(edges.size());
    for (auto& edge : edges) {
        int id1 = edge.id1 % num_rods;
        int id2 = edge.id2 % num_rods;
        if (id1 != id2) {
            VoronoiEdge e = { id1, id2 };
            // copy float 4
            __m128 pos_data = _mm_load_ps(&edge.x1);
            _mm_store_ps(&e.x1, pos_data);
            // save and roll the list
            result.push_back(e);
        }
    }
    return result;
}

vector<DelaunayUnit> TrueDelaunayModulo(const vector<VoronoiEdge>& edges, int num_rods)
{
    vector<DelaunayUnit> result(num_rods);
    for (auto& edge : edges) {
        int id1 = edge.id1 % num_rods;
        int id2 = edge.id2 % num_rods;
        if (id1 != id2) {
            if (id1 > id2)swap(id1, id2);
            int flag = 0;
            for (auto& d_unit : result[id1]) {
                if (d_unit.first == id2) {
                    // if there is already an edge, ignore it
                    flag = 1; break;
                }
                else {
                    continue;
                }
            }
            if (flag == 0) {
                result[id1].push_back({ id2,1 });
            }
        }
    }
    return result;
}


vector<DelaunayUnit> WeightedDelaunayModulo(const vector<VoronoiEdge>& edges, int num_rods)
{
    vector<DelaunayUnit> result(num_rods);
    for (auto& edge : edges) {
        int id1 = edge.id1 % num_rods;
        int id2 = edge.id2 % num_rods;
        if (id1 != id2) {
            if (id1 > id2)swap(id1, id2);
            int flag = 0;
            for (auto& d_unit : result[id1]) {
                if (d_unit.first == id2) {
                    d_unit.second += length(edge);
                    flag = 1; break;
                }
                else {
                    continue;
                }
            }
            if (flag == 0) {
                result[id1].push_back({ id2,length(edge)});
            }
        }
    }
    return result;
}
