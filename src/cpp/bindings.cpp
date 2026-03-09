#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
    py::object seed_points_obj
);

py::dict flood_fill_collect_seeds(
    py::array_t<float, py::array::c_style | py::array::forcecast> property_map,   // (Z,Y,X,C)
    py::array_t<bool,  py::array::c_style | py::array::forcecast> footprint,      // (FZ,FY,FX)
    float local_threshold,
    float footprint_tolerance,
    py::object mask_obj,
    int max_iterations,
    int min_grain_size,
);

PYBIND11_MODULE(flood_fill, m) {
    m.def("flood_fill_random_seeds_3d",
        &flood_fill_random_seeds_3d,
        py::arg("property_map"),
        py::arg("footprint"),
        py::arg("local_threshold"),
        py::arg("footprint_tolerance"),
        py::arg("mask"),
        py::arg("max_iterations"),
        py::arg("min_grain_size"),
        py::arg("recycle_small_grains"),
        py::arg("stagnation_tolerance"),
        py::arg("seed_points") = py::none()
    );
    m.def("flood_fill_collect_seeds", &flood_fill_collect_seeds, "3D flood fill with returned seeds and sizes");
}