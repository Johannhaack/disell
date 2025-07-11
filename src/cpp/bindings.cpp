#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations (match signatures exactly)


py::array_t<bool> flood_fill_incremental_mean_3D_optimized(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    float footprint_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
);

// NEW FUNCTION: your 2D multichannel flood fill
py::tuple flood_fill_2D_multichannel(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
);

PYBIND11_MODULE(_flood_fill, m) {
    m.def("flood_fill_incremental_mean_3D_optimized", &flood_fill_incremental_mean_3D_optimized, "Flood fill with connectivity");
    m.def("flood_fill_2D_multichannel", &flood_fill_2D_multichannel, "Multichannel 2D flood fill");
}
