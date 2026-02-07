#ifndef PCH_H
#define PCH_H

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#include "framework.h"
#else
#define DLLEXPORT extern "C"
#endif

DLLEXPORT int DelaunayModulo(int n, int m, int N, void* indices_in_ptr, void* edges_in_ptr, void* mask_ptr, void* indices_out_ptr,
    void* edges_out_ptr, void* weights_out_ptr);
DLLEXPORT void ConvexHull(void* points_ptr, void* out_mask_ptr, int n_points, int n_rods);
DLLEXPORT void DistToEllipse(float a, float b, void* points_ptr, void* out_ptr, int N);
DLLEXPORT void RemoveBadBoundaryEdges(void* points_ptr, void* convex_hull_ptr, void* table_ptr, void* indices_ptr, void* edges_ptr,
    void* mask_ptr, int convex_hull_length, float cos_threshold);
DLLEXPORT void neighbors(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* output_ptr);
DLLEXPORT void symmetricSum(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* a_ptr, void* output_ptr);
DLLEXPORT void complexSum(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* complex_ptr, void* output_ptr);
DLLEXPORT void symmetricMax(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    void* a_ptr, void* output_ptr);
DLLEXPORT void z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, float p);
DLLEXPORT void orientation_diff_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* configuration_ptr, void* output_ptr);
DLLEXPORT void pure_rotation_direction_phi(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_ptr);
DLLEXPORT void anisotropic_z_ij_power_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* orientation_ptr, void* output_complex_ptr, float gamma, float p);
DLLEXPORT void sumAnisotropicComplex(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, 
    void* complex_ptr, void* output_ptr);
DLLEXPORT float mean_r_ij(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr);
DLLEXPORT float segment_dist_moment(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    float gamma, int moment);
DLLEXPORT void SegmentDistForBonds(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_ptr, float gamma);
DLLEXPORT float RijRatio(void* p_xyt, int N);
DLLEXPORT int isOutOfBoundary(void* p_xyt, int N, float A, float B);
DLLEXPORT float CubicMinimum(float a, float b, float c, float d);
DLLEXPORT void convertXY(int edge_type, float r, float t1, float t2, void* xyxy_ptr);
DLLEXPORT void FittedEllipticPhi_p(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* output_complex_ptr, void* out_gamma, void* out_theta, float p);
DLLEXPORT void correlation(void* xyt_ptr, void* opA_field_ptr, void* opB_field_ptr, void* out_r_ptr, void* out_corr_ptr, int if_seg_dist,
    int N, float gamma, float mean_A, float mean_B, float std_A, float std_B);
DLLEXPORT void angularCorrelation(void* xyt_ptr, void* out_r_ptr, void* out_corr_ptr, int if_seg_dist, int N, float gamma);
DLLEXPORT void Angle57Hist(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr,
    int n_angles, void* xyt_ptr, void* z_number_ptr, void* output_ptr);
DLLEXPORT void is_isolated_defect(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* output_ptr);
DLLEXPORT void windingNumber2(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* configuration_ptr,
    void* angle_ptr, void* output_ptr);
DLLEXPORT void bitmap_from_delaunay(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* dst_ptr);
DLLEXPORT void bitmap_subtract(void* a_ptr, void* b_ptr, void* dst_ptr, int num_bytes);
DLLEXPORT int bitmap_to_pairs(void* src_ptr, void* dst_ptr, int num_rods);
DLLEXPORT int bitmap_count(void* data_ptr, int num_bytes);
DLLEXPORT void BoundaryMask(void* data_ptr, void* dst_ptr, int num_rods);
DLLEXPORT int num_rod_required_for_bitmap();
DLLEXPORT int FindEventsInBitmap(int num_rods, void* current_bonds_ptr, void* new_bonds_ptr, void* previous_z, void* current_z,
    void* dst_ptr);
DLLEXPORT void vote(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* values_ptr, void* output_ptr, int max_value);

#endif //PCH_H
