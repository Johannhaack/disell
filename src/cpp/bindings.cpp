#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations (match signatures exactly)


py::tuple flood_fill_3d_dfxm(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    float footprint_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
);

// NEW FUNCTION: your 2D multichannel flood fill
py::tuple flood_fill_2d_dfxm(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,
    std::tuple<int, int> seed_point,
    py::array_t<bool, py::array::c_style | py::array::forcecast> footprint,
    float local_tolerance,
    float global_tolerance,
    float footprint_tolerance,
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask
);

PYBIND11_MODULE(_flood_fill, m) {
    m.def("flood_fill_3d_dfxm", &flood_fill_3d_dfxm, "3D flood fill");
    m.def("flood_fill_2d_dfxm", &flood_fill_2d_dfxm, "2D flood fill");
}
