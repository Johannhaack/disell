#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <cmath>

namespace py = pybind11;


py::tuple flood_fill_2d_dfxm(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    float footprint_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
) {
    // Get array info
    auto prop_buf = property_map.request();
    auto foot_buf = footprint.request();
    auto mask_buf = mask.request();
    
    if (prop_buf.ndim != 3) {
        throw std::runtime_error("Property map must be 3D (H, W, C)");
    }
    if (foot_buf.ndim != 2) {
        throw std::runtime_error("Footprint must be 2D");
    }
    if (mask_buf.ndim != 2) {
        throw std::runtime_error("Mask must be 2D");
    }
    
    const int H = prop_buf.shape[0];
    const int W = prop_buf.shape[1];
    const int C = prop_buf.shape[2];
    const int foot_h = foot_buf.shape[0];
    const int foot_w = foot_buf.shape[1];
    const int m = foot_h / 2;
    const int n = foot_w / 2;
    
    // Get data pointers
    float* prop_ptr = static_cast<float*>(prop_buf.ptr);
    bool* foot_ptr = static_cast<bool*>(foot_buf.ptr);
    bool* mask_ptr = static_cast<bool*>(mask_buf.ptr);
    
    // Create output array
    auto result = py::array_t<bool>({H, W});
    auto res_buf = result.request();
    bool* res_ptr = static_cast<bool*>(res_buf.ptr);
    
    // Initialize result to false
    std::fill(res_ptr, res_ptr + H * W, false);
    
    const int i = std::get<0>(seed_point);
    const int j = std::get<1>(seed_point);
    
    // Check bounds and mask
    if (i < 0 || i >= H || j < 0 || j >= W || !mask_ptr[i * W + j]) {
        return result;
    }
    
    // Initialize flood fill
    res_ptr[i * W + j] = true;
    std::vector<double> total(C, 0.0);
    std::vector<double> mean_val(C, 0.0);
    
    // Initialize with seed point values
    for (int c = 0; c < C; ++c) {
        total[c] = static_cast<double>(prop_ptr[i * W * C + j * C + c]);
        mean_val[c] = total[c];
    }
    
    int count = 1;
    std::queue<std::pair<int, int>> queue;
    queue.push({i, j});

    // Preallocate containers for group logic
    std::vector<std::pair<int, int>> inner_queue;
    std::vector<double> total_inner(C, 0.0);
    std::vector<double> mean_inner(C, 0.0);
    std::vector<double> center_val(C, 0.0);

    while (!queue.empty()) {
        auto [curr_i, curr_j] = queue.front();
        queue.pop();

        // Reset containers
        inner_queue.clear();
        std::fill(total_inner.begin(), total_inner.end(), 0.0);
        mean_inner = mean_val;
        int count_inner = 0;
        int footprint_valid_voxels = 0;

        // Load center values once
        for (int c = 0; c < C; ++c)
            center_val[c] = static_cast<double>(prop_ptr[curr_i * W * C + curr_j * C + c]);

        for (int dk = 0; dk < foot_h; ++dk) {
            for (int dl = 0; dl < foot_w; ++dl) {
                if (!foot_ptr[dk * foot_w + dl]) continue;

                int row = curr_i - m + dk;
                int col = curr_j - n + dl;

                if (row >= 0 && row < H && col >= 0 && col < W) {
                    int idx = row * W + col;
                    if (!res_ptr[idx] && mask_ptr[idx]) {
                        footprint_valid_voxels++;
                        // Compute differences
                        double local_diff = 0.0, global_diff = 0.0;
                        for (int c = 0; c < C; ++c) {
                            double test_val = static_cast<double>(prop_ptr[row * W * C + col * C + c]);
                            local_diff += std::abs(test_val - center_val[c]);
                        }
                        local_diff /= C;

                        for (int c = 0; c < C; ++c) {
                            double test_val = static_cast<double>(prop_ptr[row * W * C + col * C + c]);
                            global_diff += std::abs(test_val - mean_inner[c]);
                        }
                        global_diff /= C;

                        if (local_diff < local_tolerance && global_diff < global_tolerance) {
                            inner_queue.emplace_back(row, col);
                            for (int c = 0; c < C; ++c)
                                total_inner[c] += static_cast<double>(prop_ptr[row * W * C + col * C + c]);
                            count_inner++;
                        }
                    }
                }
            }
        }

        if (footprint_valid_voxels > 0 && static_cast<float>(count_inner) / footprint_valid_voxels > footprint_tolerance) {
            // Accept group: add to queue and update stats
            for (const auto& [row, col] : inner_queue) {
                int idx = row * W + col;
                res_ptr[idx] = true;
                queue.push({row, col});
            }
            for (int c = 0; c < C; ++c) total[c] += total_inner[c];
            count += count_inner;
            for (int c = 0; c < C; ++c) mean_val[c] = total[c] / count;
        }
    }

    // Convert mean_val vector to py::array
    auto mean_orientation = py::array_t<double>(C);
    auto mean_buf = mean_orientation.request();
    double* mean_ptr = static_cast<double*>(mean_buf.ptr);

    for (int c = 0; c < C; ++c) {
        mean_ptr[c] = mean_val[c];
    }

    // Return tuple: (mask, mean orientation)
    return py::make_tuple(result, mean_orientation);
}