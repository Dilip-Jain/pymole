#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gradient.h"
#include "divergence.h"
#include "laplacian.h"

namespace py = pybind11;

template<typename T>
py::array_t<double> sparse_to_numpy(const T& mat) {
    // Convert sparse matrix to CSC format numpy array
    // This will need implementation based on MOLE's matrix format
    // For now this is a placeholder
    return py::array_t<double>();
}

PYBIND11_MODULE(_operators, m) {
    // Gradient operator
    py::class_<Gradient>(m, "GradientOperator")
        .def(py::init<int, double>())
        .def("periodic", &Gradient::periodic)
        .def("nonperiodic", &Gradient::nonperiodic)
        .def("matrix", [](const Gradient& op) {
            return sparse_to_numpy(op.matrix());
        });

    // Divergence operator
    py::class_<Divergence>(m, "DivergenceOperator")
        .def(py::init<int, double>())
        .def("periodic", &Divergence::periodic)
        .def("nonperiodic", &Divergence::nonperiodic)
        .def("matrix", [](const Divergence& op) {
            return sparse_to_numpy(op.matrix());
        });

    // Laplacian operator
    py::class_<Laplacian>(m, "LaplacianOperator")
        .def(py::init<int, double>())
        .def("periodic", &Laplacian::periodic)
        .def("nonperiodic", &Laplacian::nonperiodic)
        .def("matrix", [](const Laplacian& op) {
            return sparse_to_numpy(op.matrix());
        });
}
