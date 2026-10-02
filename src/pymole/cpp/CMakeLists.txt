cmake_minimum_required(VERSION 3.15)
project(pymole_cpp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically prepare MOLE dependencies before building
set(MOLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
set(ARMADILLO_SOURCE_DIR "${MOLE_SOURCE_DIR}/armadillo-12.6.6")
if(NOT EXISTS "${MOLE_SOURCE_DIR}/gradient.h" OR NOT EXISTS "${ARMADILLO_SOURCE_DIR}/build/tmp/include/armadillo")
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
endif()

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add MOLE source files (only once)
file(GLOB MOLE_SOURCES
    ${MOLE_SOURCE_DIR}/*.cpp
)

# Build the Python extension
pybind11_add_module(_operators 
    operators.cpp
    ${MOLE_SOURCES}
    ${ARMADILLO_SOURCE_DIR}/src/wrapper1.cpp
    ${ARMADILLO_SOURCE_DIR}/src/wrapper2.cpp
)

# Add include directories
target_include_directories(_operators PRIVATE
    ${MOLE_SOURCE_DIR}
    ${ARMADILLO_SOURCE_DIR}/build/tmp/include
)

if(MINGW)
    target_compile_options(_operators PRIVATE -mcrtdll=ucrt)
    target_link_options(_operators PRIVATE -mcrtdll=ucrt -static-libgcc -static-libstdc++)
    get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    install(FILES "${MINGW_BIN_DIR}/libwinpthread-1.dll" DESTINATION pymole/cpp)
endif()

# Install the extension module
install(TARGETS _operators LIBRARY DESTINATION pymole/cpp RUNTIME DESTINATION pymole/cpp)
