cmake_minimum_required(VERSION 3.15)
project(flood_fill LANGUAGES CXX)

find_package(pybind11 REQUIRED)

# Add your extension module, including all relevant sources
pybind11_add_module(_flood_fill
    bindings.cpp
    flood_fill_2d.cpp
    flood_fill_3d.cpp
)

# Install the built .so into your Python package folder
install(TARGETS _flood_fill
    LIBRARY DESTINATION disell
)
