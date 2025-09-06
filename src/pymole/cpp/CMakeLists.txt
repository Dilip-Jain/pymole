cmake_minimum_required(VERSION 3.15)
project(pymole_cpp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add MOLE source files
file(GLOB MOLE_SOURCES "src/cpp/*.cpp")

# Add the pybind11 module
pybind11_add_module(_operators 
    operators.cpp
    ${MOLE_SOURCES}
)

# Include directories (headers live in src/cpp)
target_include_directories(_operators PRIVATE
    src/cpp
)
