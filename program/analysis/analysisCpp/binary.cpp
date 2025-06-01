#include "pch.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <immintrin.h>

using namespace std;

struct BitmapRef {
    uint8_t* data;
    int num_bytes;
    int bytes_per_row;

    BitmapRef(uint8_t* data_ptr, int num_rods) : data(data_ptr) {
        num_bytes = num_rods * num_rods / 8;
        bytes_per_row = num_rods / 8;
    }

    // Set individual bit value to 1
    void set(int i, int j) {
        int pos = i * bytes_per_row * 8 + j;
        int byte_idx = pos / 8;
        data[byte_idx] |= (uint8_t)1 << (pos % 8);
    }

    // Row-wise OR operation: returns vector of row indices that contain at least one 1
    std::vector<int> row_wise_or() const {
        const int rows = num_bytes / bytes_per_row;
        std::vector<int> result;
        result.reserve(rows);  // Reserve worst-case space

        for (int r = 0; r < rows; ++r) {
            uint8_t* row_start = data + r * bytes_per_row;
            bool row_has_one = false;
            int bytes_processed = 0;

            // Process 32-byte chunks with AVX2
            while (bytes_processed + 32 <= bytes_per_row) {
                __m256i vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_start + bytes_processed));

                if (!_mm256_testz_si256(vec, vec)) {
                    row_has_one = true;
                    break;  // Short-circuit on first non-zero chunk
                }
                bytes_processed += 32;
            }
            // Process remaining bytes if needed
            if (!row_has_one) {
                for (int i = bytes_processed; i < bytes_per_row; ++i) {
                    if (row_start[i] != 0) {
                        row_has_one = true;
                        break;
                    }
                }
            }
            // If row has at least one 1, record the row index
            if (row_has_one) {
                result.push_back(r);
            }
        }
        return result;
    }

    // Find connected components in the subset of nodes defined by indices
    std::vector<std::vector<int>> cluster(const std::vector<int>& indices) const {
        const int n = indices.size();
        if (n == 0) return {};

        // Create mapping from original index to compressed index
        std::unordered_map<int, int> index_map;
        for (int i = 0; i < n; i++) {
            index_map[indices[i]] = i;
        }

        // Initialize Union-Find data structure
        std::vector<int> parent(n);
        std::vector<int> rank(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }

        // Find with path compression
        auto find = [&](int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];  // Path compression
                x = parent[x];
            }
            return x;
            };

        // Union by rank
        auto union_sets = [&](int x, int y) {
            x = find(x);
            y = find(y);
            if (x == y) return;

            if (rank[x] < rank[y]) {
                parent[x] = y;
            }
            else if (rank[x] > rank[y]) {
                parent[y] = x;
            }
            else {
                parent[y] = x;
                rank[x]++;
            }
            };

        // Precompute a bitmask for the indices set
        std::vector<uint8_t> in_indices(bytes_per_row, 0);
        for (int idx : indices) {
            int byte_idx = idx / 8;
            int bit_offset = idx % 8;
            in_indices[byte_idx] |= (1 << bit_offset);
        }

        // Process relationships in batches using SIMD
        for (int i = 0; i < n; i++) {
            int orig_i = indices[i];
            const uint8_t* row = data + orig_i * bytes_per_row;

            // Process 32-byte chunks
            int byte_idx = 0;
            for (; byte_idx + 32 <= bytes_per_row; byte_idx += 32) {
                __m256i row_chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row + byte_idx));
                __m256i indices_chunk = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(in_indices.data() + byte_idx));
                __m256i connected = _mm256_and_si256(row_chunk, indices_chunk);

                if (_mm256_testz_si256(connected, connected)) {
                    continue;  // No connections in this chunk
                }

                // Process each byte in the chunk
                uint8_t* connected_bytes = reinterpret_cast<uint8_t*>(&connected);
                for (int j = 0; j < 32; j++) {
                    uint8_t byte_val = connected_bytes[j];
                    if (!byte_val) continue;

                    int base_col = (byte_idx + j) * 8;
                    for (int bit = 0; bit < 8; bit++) {
                        if (byte_val & (1 << bit)) {
                            int orig_j = base_col + bit;
                            if (auto it = index_map.find(orig_j); it != index_map.end()) {
                                int compressed_j = it->second;
                                if (i != compressed_j) {  // Avoid self-loop
                                    union_sets(i, compressed_j);
                                }
                            }
                        }
                    }
                }
            }

            // Process remaining bytes
            for (; byte_idx < bytes_per_row; byte_idx++) {
                uint8_t byte_val = row[byte_idx] & in_indices[byte_idx];
                if (!byte_val) continue;

                int base_col = byte_idx * 8;
                for (int bit = 0; bit < 8; bit++) {
                    if (byte_val & (1 << bit)) {
                        int orig_j = base_col + bit;
                        if (auto it = index_map.find(orig_j); it != index_map.end()) {
                            int compressed_j = it->second;
                            if (i != compressed_j) {  // Avoid self-loop
                                union_sets(i, compressed_j);
                            }
                        }
                    }
                }
            }
        }

        // Group nodes by their root parent
        std::unordered_map<int, std::vector<int>> clusters;
        for (int i = 0; i < n; i++) {
            int root = find(i);
            clusters[root].push_back(indices[i]);  // Store original index
        }

        // Convert to result format
        std::vector<std::vector<int>> result;
        result.reserve(clusters.size());
        for (auto& [root, nodes] : clusters) {
            result.push_back(std::move(nodes));
        }

        return result;
    }
};

// Convert delaunay indices to bitmap
void bitmap_from_delaunay(int num_edges, int num_rods, void* indices_ptr, void* edges_ptr, void* dst_ptr) {
    BitmapRef dst = BitmapRef((uint8_t*)dst_ptr, num_rods);
    int* indices = (int*)indices_ptr;
    int* edges = (int*)edges_ptr;
    
    int id1 = 0;
    for (int j = 0; j < num_edges; j++) {
        while (j == *indices && id1 < num_rods) {
            indices++;
            id1++;
        }
        int id2 = edges[j];
        dst.set(id1, id2); dst.set(id2, id1);
    }
}

// SIMD-accelerated vector subtraction: a - b (1-1=0, 1-0=1, 0-0=0, 0-1=0)
void bitmap_subtract(void* a_ptr, void* b_ptr, void* dst_ptr, int num_bytes) {
    uint8_t* a = static_cast<uint8_t*>(a_ptr);
    uint8_t* b = static_cast<uint8_t*>(b_ptr);
    uint8_t* dst = static_cast<uint8_t*>(dst_ptr);

    // Process 32 bytes (256 bits) per iteration
    for (size_t i = 0; i < num_bytes; i += 32) {
        // Load 256-bit chunks
        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        // Compute a & ~b using AVX2 (rule: 1-1=0, 1-0=1, 0-0=0, 0-1=0)
        __m256i res = _mm256_andnot_si256(b_vec, a_vec);

        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), res);
    }
}

int bitmap_count(void* data_ptr, int num_bytes) {
    // 256-bit lookup table
    // Generated by python code: str([bin(x).count('1') for x in range(256)])
    constexpr int nibble_lookup[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    };

    uint8_t* data = static_cast<uint8_t*>(data_ptr);
    int total = 0;
    for (int i = 0; i < num_bytes; i++) {
        total += nibble_lookup[data[i]];
    }
    return total;
}

int bitmap_to_pairs(void* src_ptr, void* dst_ptr, int num_rods) {
    BitmapRef src = BitmapRef((uint8_t*)src_ptr, num_rods);
    std::pair<int, int>* dst = static_cast<std::pair<int, int>*>(dst_ptr);
    std::pair<int, int>* moving_dst = static_cast<std::pair<int, int>*>(dst_ptr);
    
    for (int i = 0; i < src.num_bytes; i++) {
        int id1 = i / src.bytes_per_row;
        for (int j = 0; j < 8; j++) {
            if (src.data[i] & (1 << j)) {
                int id2 = j + (i % src.bytes_per_row) * 8;
                if (id1 < id2) *moving_dst++ = { id1, id2 };
            }
        }
    }
    return moving_dst - dst;
}

struct DefectEvent {
    int related_particles;
    int previous_negative_charge = 0, previous_positive_charge = 0;
    int current_negative_charge = 0, current_positive_charge = 0;

    DefectEvent(vector<int>& particles, int* previous_z, int* current_z) {
        related_particles = particles.size();
        int charge;
        for (int id : particles) {
            charge = previous_z[id] - 6;
            if (charge < 0) previous_negative_charge -= charge;
            else if (charge > 0) previous_positive_charge += charge;
            charge = current_z[id] - 6;
            if (charge < 0) current_negative_charge -= charge;
            else if (charge > 0) current_positive_charge += charge;
        }
    }
};

vector<DefectEvent> clusterByFullGraph(BitmapRef& current_bonds, BitmapRef& new_bonds, int* previous_z, int* current_z) {
    vector<int> indices = new_bonds.row_wise_or();
    vector<vector<int>> clusters = current_bonds.cluster(indices);
    vector<DefectEvent> events;
    for(auto& cluster : clusters) {
        events.emplace_back(cluster, previous_z, current_z);
    }
    return events;
}

int FindEventsInBitmap(int num_rods, void* current_bonds_ptr, void* new_bonds_ptr, void* previous_z, void* current_z, 
    void* dst_ptr) 
{
    BitmapRef current_bonds = BitmapRef((uint8_t*)current_bonds_ptr, num_rods);
    BitmapRef new_bonds = BitmapRef((uint8_t*)new_bonds_ptr, num_rods);
    vector<DefectEvent> events = clusterByFullGraph(current_bonds, new_bonds, (int*)previous_z, (int*)current_z);
    memcpy(dst_ptr, events.data(), events.size() * sizeof(DefectEvent));
    return events.size();
}