#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <tuple>
#include <vector>
#include <cmath>
#include <unordered_set>

namespace py = pybind11;

// Custom hash function for 3D coordinates
struct CoordHash {
    std::size_t operator()(const std::tuple<int, int, int>& coord) const {
        auto [x, y, z] = coord;
        return std::hash<int>{}(x) ^ (std::hash<int>{}(y) << 1) ^ (std::hash<int>{}(z) << 2);
    }
};

py::tuple flood_fill_3d_dfxm(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    float footprint_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
) {
    auto buf_prop = property_map.unchecked<4>();
    auto buf_foot = footprint.unchecked<3>();
    auto buf_mask = mask.unchecked<3>();

    const int Z = buf_prop.shape(0);
    const int Y = buf_prop.shape(1);
    const int X = buf_prop.shape(2);
    const int C = buf_prop.shape(3);
    const int fz = buf_foot.shape(0);
    const int fy = buf_foot.shape(1);
    const int fx = buf_foot.shape(2);
    const int mz = fz / 2;
    const int my = fy / 2;
    const int mx = fx / 2;

    // Input validation
    auto [si, sj, sk] = seed_point;
    if (si < 0 || sj < 0 || sk < 0 || si >= Z || sj >= Y || sk >= X) {
        throw std::invalid_argument("Seed point is out of bounds");
    }

    auto flood_mask = py::array_t<bool>({Z, Y, X});
    auto buf_flood = flood_mask.mutable_unchecked<3>();

    // Initialize flood mask - more cache-friendly order
    std::fill_n(flood_mask.mutable_data(), Z * Y * X, false);

    if (!buf_mask(si, sj, sk)) return flood_mask;

    // Pre-compute footprint offsets and sum for better cache locality
    std::vector<std::tuple<int, int, int>> footprint_offsets;
    footprint_offsets.reserve(fz * fy * fx); // Pre-allocate
    
    for (int dz = 0; dz < fz; ++dz) {
        for (int dy = 0; dy < fy; ++dy) {
            for (int dx = 0; dx < fx; ++dx) {
                if (buf_foot(dz, dy, dx)) {
                    footprint_offsets.emplace_back(dz - mz, dy - my, dx - mx);
                }
            }
        }
    }

    std::queue<std::tuple<int, int, int>> queue;
    std::vector<double> total(C); // Use double for better numerical precision
    std::vector<float> mean_val(C);
    
    // Initialize with seed point values
    for (int c = 0; c < C; ++c) {
        total[c] = buf_prop(si, sj, sk, c);
        mean_val[c] = total[c];
    }
    int count = 1;

    buf_flood(si, sj, sk) = true;
    queue.emplace(si, sj, sk);

    // Pre-allocate vectors to avoid repeated allocations
    std::vector<std::tuple<int, int, int>> inner_queue;
    std::vector<double> total_inner(C);
    std::vector<float> mean_inner(C);
    std::vector<float> center_val(C);
    std::vector<float> val(C);
    
    // Reserve space to reduce allocations
    inner_queue.reserve(footprint_offsets.size());

    while (!queue.empty()) {
        auto [i, j, k] = queue.front();
        queue.pop();

        // Load center values once
        for (int c = 0; c < C; ++c) {
            center_val[c] = buf_prop(i, j, k, c);
        }

        // Reset containers
        inner_queue.clear();
        std::fill(total_inner.begin(), total_inner.end(), 0.0);
        int count_inner = 0;
        mean_inner = mean_val;
        int valid_voxels_footprint = 0;
        // Use pre-computed offsets
        for (const auto& [dz, dy, dx] : footprint_offsets) {
            int ni = i + dz;
            int nj = j + dy;
            int nk = k + dx;
            
            // Bounds check
            if (ni < 0 || nj < 0 || nk < 0 || ni >= Z || nj >= Y || nk >= X) continue;
            if (buf_flood(ni, nj, nk) || !buf_mask(ni, nj, nk))continue;
            valid_voxels_footprint += 1;
            // Load neighbor values
            float local_diff = 0.0f, global_diff = 0.0f;
            for (int c = 0; c < C; ++c) {
                val[c] = buf_prop(ni, nj, nk, c);
                local_diff += std::abs(val[c] - center_val[c]);
                global_diff += std::abs(val[c] - mean_inner[c]);
            }
            
            // Normalize differences
            const float inv_C = 1.0f / C;
            local_diff *= inv_C;
            global_diff *= inv_C;

            if (local_diff < local_tolerance && global_diff < global_tolerance) {
                inner_queue.emplace_back(ni, nj, nk);
                for (int c = 0; c < C; ++c) {
                    total_inner[c] += val[c];
                }
                ++count_inner;

                // Update mean incrementally for better numerical stability
                const double inv_total_count = 1.0 / (count_inner + count);
                for (int c = 0; c < C; ++c) {
                    mean_inner[c] = static_cast<float>((total_inner[c] + total[c]) * inv_total_count);
                }
            }
        }

        // Check footprint tolerance
        if (valid_voxels_footprint == 0 || static_cast<float>(count_inner) / valid_voxels_footprint > footprint_tolerance) {
            // Accept all points in inner_queue
            for (const auto& [ni, nj, nk] : inner_queue) {
                buf_flood(ni, nj, nk) = true;
                queue.emplace(ni, nj, nk);
            }
            
            // Update global statistics
            for (int c = 0; c < C; ++c) {
                total[c] += total_inner[c];
            }
            count += count_inner;
            
            // Update mean with better numerical precision
            const double inv_count = 1.0 / count;
            for (int c = 0; c < C; ++c) {
                mean_val[c] = static_cast<float>(total[c] * inv_count);
            }
        }
    }

    // Convert mean_val vector to py::array
    auto mean_orientation = py::array_t<double>(C);
    auto mean_buf = mean_orientation.request();
    double* mean_ptr = static_cast<double*>(mean_buf.ptr);

    for (int c = 0; c < C; ++c) {
        mean_ptr[c] = static_cast<double>(mean_val[c]);
    }

    // Return tuple: (mask, mean orientation)
    return py::make_tuple(flood_mask, mean_orientation);
}

