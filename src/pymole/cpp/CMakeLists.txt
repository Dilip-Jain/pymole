cmake_minimum_required(VERSION 3.15)
project(pymole_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add MOLE source files
set(MOLE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cpp/gradient.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/divergence.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/laplacian.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/utils.cpp
)

# Add the pybind11 module
pybind11_add_module(_operators 
    operators.cpp
    ${MOLE_SOURCES}
)

# Include directories
target_include_directories(_operators PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
)

# Set output directory
set_target_properties(_operators PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pymole/src/pymole/cpp"
)
