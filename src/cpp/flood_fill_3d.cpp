// flood_fill_batch_3d.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_set>
#include <random>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

// ------------------- offset struct -------------------
struct Offset3D {
    int dz, dy, dx;
    ptrdiff_t dlin;   // precomputed linear offset

    Offset3D(int dz_, int dy_, int dx_, int Y, int X)
        : dz(dz_), dy(dy_), dx(dx_) 
    {
        dlin = (ptrdiff_t)dz * (Y * X) + (ptrdiff_t)dy * X + dx;
    }
};



void flood_fill_single_region_binary_3d(
    const float* __restrict prop,
    const uint8_t*  __restrict mask,
    int Z, int Y, int X, int C,
    const std::vector<Offset3D>& offsets,
    int si, int sj, int sk,
    float thr_sq_C,
    float footprint_tolerance,
    std::vector<size_t>& region_indices
) {

    // bounds
    if (si < 0 || sj < 0 || sk < 0 ||
        si >= Z || sj >= Y || sk >= X) {
        region_indices.clear();
        return;
    }

    const size_t seed_idx = (size_t)si * (Y * X) + (size_t)sj * X + sk;
    if (!mask[seed_idx]) {
        region_indices.clear();
        return;
    }


    const size_t Nvox = (size_t)Z * Y * X;
    std::vector<uint8_t> visited(Nvox, 0);

    // local stack (DFS)
    std::vector<size_t> stack;
    stack.reserve(8192);
    stack.push_back(seed_idx);

    // mark seed visited
    visited[seed_idx] = 1;

    region_indices.clear();

    const size_t stride_yx = (size_t)Y * X;

    std::vector<size_t> candidates;
    candidates.reserve(offsets.size());

    while (!stack.empty()) {

        size_t idx = stack.back();
        stack.pop_back();

        int z = idx / stride_yx;
        int y = (idx % stride_yx) / X;
        int x = idx % X;

        const float* center = prop + idx*C;

        candidates.clear();
        int valid_neighbors = 0;
        int count_pass = 0;

        for (const auto& off : offsets) {
            size_t nidx = idx + off.dlin;

            if (!mask[nidx]) continue;
            valid_neighbors++;

            const float* neigh = prop + nidx*C;

            float dist2 = 0.0f;
            #pragma omp simd reduction(+:dist2)
            for (int c = 0; c < C; ++c) {
                float d = neigh[c] - center[c];
                dist2 += d*d;
            }

            bool pass = dist2 < thr_sq_C;
            count_pass += pass;
            if (pass) candidates.push_back(nidx);
        }

        // int-based threshold
        int min_pass = (int)std::ceil(footprint_tolerance * valid_neighbors);
        if (count_pass >= min_pass) {
            region_indices.push_back(idx);

            for (size_t nidx : candidates) {
                if (!visited[nidx]) {
                    visited[nidx] = 1;
                    stack.push_back(nidx);
                }
            }
        }
    }
}



// ------------------- main batch function (3D random seeds) -------------------

py::dict flood_fill_random_seeds_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,   // (Z,Y,X,C)
    py::array_t<bool,  py::array::c_style | py::array::forcecast> footprint,      // (FZ,FY,FX)
    float local_threshold,
    float footprint_tolerance,
    py::object mask_obj,
    int max_iterations,
    int min_grain_size,
    bool recycle_small_grains,
    int stagnation_tolerance,
    py::object seed_points_obj = py::none()
) {
    // ---- property_map checks ----
    auto pbuf = property_map.request();
    if (pbuf.ndim != 4)
        throw std::runtime_error("property_map must be 4D (Z,Y,X,C)");

    const int Z = pbuf.shape[0];
    const int Y = pbuf.shape[1];
    const int X = pbuf.shape[2];
    const int C = pbuf.shape[3];
    const size_t Nvox = (size_t)Z * Y * X;

    const float* prop = static_cast<const float*>(pbuf.ptr);

    // ---- footprint ----
    auto fbuf = footprint.request();
    if (fbuf.ndim != 3)
        throw std::runtime_error("footprint must be 3D (FZ,FY,FX)");
    const int FZ = fbuf.shape[0];
    const int FY = fbuf.shape[1];
    const int FX = fbuf.shape[2];
    const bool* fdata = static_cast<const bool*>(fbuf.ptr);

    const int mZ = FZ / 2;
    const int mY = FY / 2;
    const int mX = FX / 2;

    // ---- mask (now REQUIRED) ----
    if (mask_obj.is_none())
        throw std::runtime_error("mask must be provided (Python must compute it)");

    py::array_t<uint8_t> mask_arr = mask_obj.cast<py::array_t<uint8_t>>();
    auto mbuf = mask_arr.request();

    if (mbuf.ndim != 3 ||
        mbuf.shape[0] != Z ||
        mbuf.shape[1] != Y ||
        mbuf.shape[2] != X)
        throw std::runtime_error("mask must be shape (Z,Y,X) uint8");

    uint8_t* mask = static_cast<uint8_t*>(mbuf.ptr);

    // ---- precompute footprint offsets ----
    std::vector<Offset3D> offsets;
    offsets.reserve(FZ * FY * FX);
    for (int dz = 0; dz < FZ; ++dz)
        for (int dy = 0; dy < FY; ++dy)
            for (int dx = 0; dx < FX; ++dx)
                if (fdata[(dz*FY + dy)*FX + dx])
                    offsets.emplace_back(dz - mZ, dy - mY, dx - mX,Y,X);

    // ---- segmentation output ----
    py::array_t<int> seg_arr({Z, Y, X});
    int* segmentation = seg_arr.mutable_data();
    std::fill_n(segmentation, Nvox, 0);

    // ---- region stats ----
    std::vector<size_t>    label_sizes;
    std::vector<std::vector<double>> label_means;

    const float thr_sq_C = local_threshold * local_threshold * float(C);

    // rng
    std::mt19937 rng(std::random_device{}());

    int label = 1;
    int iteration = 0;
    int last_success = -1;

    std::vector<size_t> region_indices;
    region_indices.reserve(8192);

    std::vector<size_t> remaining;
    remaining.reserve(Nvox);
    std::vector<size_t> position_in_remaining(Nvox, (size_t)-1);

    for (size_t idx = 0; idx < Nvox; ++idx) {
        if (mask[idx]) {
            position_in_remaining[idx] = remaining.size();
            remaining.push_back(idx);
        }
    }

    // ---- user-provided seeds (optional) ----
    std::vector<size_t> user_seeds;

    if (!seed_points_obj.is_none()) {
        py::array_t<long long> seeds_arr = seed_points_obj.cast<py::array_t<long long>>();
        auto sbuf = seeds_arr.request();

        if (sbuf.ndim != 2 || sbuf.shape[1] != 3)
            throw std::runtime_error("seed_points must be array of shape (N, 3)");

        long long* sptr = static_cast<long long*>(sbuf.ptr);
        size_t Nseeds = sbuf.shape[0];

        user_seeds.reserve(Nseeds);

        for (size_t i = 0; i < Nseeds; ++i) {
            long long z = sptr[i*3 + 0];
            long long y = sptr[i*3 + 1];
            long long x = sptr[i*3 + 2];

            size_t idx = (size_t)z * (Y * X) + (size_t)y * X + (size_t)x;

            // Only accept seeds that are inside the mask
            if (mask[idx]){
                user_seeds.push_back(idx);
            }
            else{
                py::print("user seed:", z, y, x, "is not in mask");
            }
        }
    }


    auto remove_voxel = [&](size_t idx_to_remove) {
        size_t pos = position_in_remaining[idx_to_remove];
        if (pos == (size_t)-1) return;  // already removed

        size_t last_idx = remaining.back();

        remaining[pos] = last_idx;
        position_in_remaining[last_idx] = pos;

        remaining.pop_back();
        position_in_remaining[idx_to_remove] = (size_t)-1;
    };

    // ================================================================
    //               MAIN LOOP (unchanged except mask logic)
    // ================================================================
    while (iteration < max_iterations) {
        if (remaining.empty())
            break;

        size_t seed_idx;

        // ---- deterministic mode: only use user seeds, stop when empty ----
        if (!user_seeds.empty()) {
            seed_idx = user_seeds.back();
            user_seeds.pop_back();

            // If user seeds become empty → stop immediately
            if (user_seeds.empty())
                max_iterations = iteration + 1;  // ensure loop exits after this iteration
        }
        // ---- random mode (no user seeds provided) ----
        else if (seed_points_obj.is_none()) {
            if (remaining.empty())
                break;

            std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
            size_t pool_pos = dist(rng);
            seed_idx = remaining[pool_pos];
        }

        int z = seed_idx / (Y * X);
        int y = (seed_idx / X) % Y;
        int x = seed_idx % X;
        // run region grow
        flood_fill_single_region_binary_3d(
            prop,
            mask,
            Z, Y, X, C,
            offsets,
            z, y, x,
            thr_sq_C,
            footprint_tolerance,
            region_indices
        );

        size_t grain_size = region_indices.size();
        if (grain_size == 0) { iteration++; continue; }

        // mean feature
        std::vector<double> mean_feat(C, 0.0);
        for (size_t idx : region_indices) {
            const float* f = &prop[idx * C];
            for (int c = 0; c < C; ++c) mean_feat[c] += f[c];
        }
        for (int c = 0; c < C; ++c)
            mean_feat[c] /= double(grain_size);

        // ===============================================================
        //                   MERGING LOGIC (unchanged)
        // ===============================================================
        // ---- MERGING / NEW-LABEL LOGIC ----
        if (grain_size <= (size_t)min_grain_size) {
            iteration++;
        }
        // Case 2: LARGE region → assign a new label
        else {

            int new_label = label++;

            for (size_t idx : region_indices) {
                segmentation[idx] = new_label;
                mask[idx] = false;        // always remove from mask
                remove_voxel(idx);
            }

            // store statistics
            label_sizes.push_back(grain_size);
            label_means.push_back(mean_feat);

            last_success = iteration;
        }
        iteration++;

        if (stagnation_tolerance > 0 &&
            (iteration - last_success) > stagnation_tolerance)
            break;

    }

    // ===============================================================
    //        RELABEL + BUILD OUTPUT (unchanged)
    // ===============================================================
    int num_labels_raw = label - 1;
    std::vector<int> new_map(num_labels_raw, 0);
    int new_id = 1;

    for (int lbl = 1; lbl <= num_labels_raw; ++lbl)
        if ((int)label_sizes[lbl-1] >= min_grain_size)
            new_map[lbl-1] = new_id++;

    int num_final = new_id - 1;

    // relabel
    for (size_t idx = 0; idx < Nvox; ++idx) {
        int old = segmentation[idx];
        if (old > 0)
            segmentation[idx] = new_map[old-1];
    }

    // outputs
    py::array_t<double> means_arr({num_final, C});
    py::array_t<long long> sizes_arr({num_final});

    auto mp = static_cast<double*>(means_arr.mutable_data());
    auto sp = static_cast<long long*>(sizes_arr.mutable_data());

    for (int lbl = 1; lbl <= num_labels_raw; ++lbl) {
        int new_lbl = new_map[lbl-1];
        if (new_lbl == 0) continue;
        int out = new_lbl - 1;
        sp[out] = label_sizes[lbl-1];
        for (int c = 0; c < C; ++c)
            mp[out*C + c] = label_means[lbl-1][c];
    }

    py::dict out;
    out["segmentation"] = seg_arr;
    out["means"]        = means_arr;
    out["sizes"]        = sizes_arr;
    return out;
}


py::dict flood_fill_collect_seeds(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,   // (Z,Y,X,C)
    py::array_t<bool,  py::array::c_style | py::array::forcecast> footprint,      // (FZ,FY,FX)
    float local_threshold,
    float footprint_tolerance,
    py::object mask_obj,
    int max_iterations,
    int min_grain_size
) {
    // ---- property_map checks ----
    auto pbuf = property_map.request();
    if (pbuf.ndim != 4)
        throw std::runtime_error("property_map must be 4D (Z,Y,X,C)");

    const int Z = pbuf.shape[0];
    const int Y = pbuf.shape[1];
    const int X = pbuf.shape[2];
    const int C = pbuf.shape[3];
    const size_t Nvox = (size_t)Z * Y * X;
    const float* prop = static_cast<const float*>(pbuf.ptr);

    std::vector<uint8_t> visited(Nvox, 0);

    // ---- footprint ----
    auto fbuf = footprint.request();
    if (fbuf.ndim != 3)
        throw std::runtime_error("footprint must be 3D (FZ,FY,FX)");
    const int FZ = fbuf.shape[0];
    const int FY = fbuf.shape[1];
    const int FX = fbuf.shape[2];
    const bool* fdata = static_cast<const bool*>(fbuf.ptr);

    const int mZ = FZ / 2;
    const int mY = FY / 2;
    const int mX = FX / 2;

    // ---- mask ----
    if (mask_obj.is_none())
        throw std::runtime_error("mask must be provided");

    py::array_t<uint8_t> mask_arr = mask_obj.cast<py::array_t<uint8_t>>();
    auto mbuf = mask_arr.request();

    if (mbuf.ndim != 3 ||
        mbuf.shape[0] != Z || mbuf.shape[1] != Y || mbuf.shape[2] != X)
        throw std::runtime_error("mask must be shape (Z,Y,X)");

    uint8_t* mask = static_cast<uint8_t*>(mbuf.ptr);

    // ---- precompute offsets ----
    std::vector<Offset3D> offsets;
    offsets.reserve(FZ * FY * FX);
    for (int dz = 0; dz < FZ; ++dz)
        for (int dy = 0; dy < FY; ++dy)
            for (int dx = 0; dx < FX; ++dx)
                if (fdata[(dz*FY + dy)*FX + dx])
                    offsets.emplace_back(dz - mZ, dy - mY, dx - mX, Y, X);

    // ---- seed candidates ----
    std::vector<size_t> remaining;
    remaining.reserve(Nvox);
    for (size_t idx = 0; idx < Nvox; ++idx)
        if (mask[idx])
            remaining.push_back(idx);

    // ---- output storage ----
    std::vector<long long> region_sizes;
    std::vector<long long> seed_points;
    seed_points.reserve(max_iterations * 3);
    region_sizes.reserve(max_iterations);


    const float thr_sq_C = local_threshold * local_threshold * (float)C;

    // RNG
    std::mt19937 rng(std::random_device{}());

    std::vector<size_t> region_indices;
    region_indices.reserve(8192);

    int iteration = 0;
    int last_success = -1;

    // ---- main loop ----
    while (iteration < max_iterations) {

        if (remaining.empty())
            break;

        std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
        size_t pool_pos = dist(rng);
        size_t seed_idx = remaining[pool_pos];

        int z = seed_idx / (Y * X);
        int y = (seed_idx / X) % Y;
        int x = seed_idx % X;

        // flood fill
        flood_fill_single_region_binary_3d(
            prop, mask, Z, Y, X, C,
            offsets,
            z, y, x,
            thr_sq_C,
            footprint_tolerance,
            region_indices
        );

        size_t grain_size = region_indices.size();
        if (grain_size == 0) {
            iteration++;
            continue;
        }

        // ---- store only size + seed ----
        if ((int)grain_size >= min_grain_size) {
            region_sizes.push_back((long long)grain_size);
            seed_points.push_back((long long)z);
            seed_points.push_back((long long)y);
            seed_points.push_back((long long)x);
        }

        iteration++;

    }

    size_t num_regions = region_sizes.size();
    
    py::array_t<long long> sizes_arr((py::ssize_t)num_regions);

    py::array_t<long long> seeds_arr(py::array::ShapeContainer{
        (py::ssize_t)num_regions,
        (py::ssize_t)3
    });


    std::memcpy(
        sizes_arr.mutable_data(),
        region_sizes.data(),
        num_regions * sizeof(long long)
    );

    std::memcpy(
        seeds_arr.mutable_data(),
        seed_points.data(),
        num_regions * 3 * sizeof(long long)
    );

    py::dict out;
    out["sizes"] = sizes_arr;
    out["seeds"] = seeds_arr;
    return out;
}
