cmake_minimum_required(VERSION 3.15)
project(pymole_cpp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically prepare MOLE dependencies before building
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "
import sys; sys.path.insert(0, '${CMAKE_CURRENT_SOURCE_DIR}/../../..');
from build_tools.prepare_mole import prepare_mole;
from pathlib import Path;
prepare_mole(Path('${CMAKE_CURRENT_SOURCE_DIR}/../../..'))
"
    RESULT_VARIABLE PREPARE_RESULT
    ERROR_VARIABLE PREPARE_ERROR
)

if(NOT PREPARE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to prepare MOLE dependencies:\n${PREPARE_ERROR}")
endif()

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

# Link Armadillo wrapper objects directly (embeds them in the .pyd)
target_link_libraries(_operators PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/armadillo-12.6.6/build/CMakeFiles/armadillo.dir/src/wrapper1.cpp.obj"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/armadillo-12.6.6/build/CMakeFiles/armadillo.dir/src/wrapper2.cpp.obj"
)

# Try to link static Armadillo library if available
find_library(ARMADILLO_LIB 
    NAMES armadillo
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/armadillo-12.6.6/build"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(ARMADILLO_LIB)
    target_link_libraries(_operators PRIVATE ${ARMADILLO_LIB})
    message(STATUS "Linked static Armadillo library: ${ARMADILLO_LIB}")
else()
    message(STATUS "Static Armadillo library not found, relying on object files")
endif()

# Install the extension module
install(TARGETS _operators LIBRARY DESTINATION pymole/cpp)
