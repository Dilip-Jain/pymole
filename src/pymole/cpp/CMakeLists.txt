cmake_minimum_required(VERSION 3.15)
project(pymole_cpp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add MOLE source files (only once)
file(GLOB MOLE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp
)

# Build the Python extension
pybind11_add_module(_operators 
    operators.cpp
    ${MOLE_SOURCES}
)


# Add include directories
target_include_directories(_operators PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/armadillo-12.6.6/include
)
