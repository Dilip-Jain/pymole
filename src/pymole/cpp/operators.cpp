#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define ARMA_USE_OPENMP
#define ARMA_USE_SUPERLU
#include <armadillo>

#include <gradient.h>
#include <divergence.h>
#include <laplacian.h>

namespace py = pybind11;

template<typename T>
py::dict sparse_to_numpy(const T& mat) {
    // Convert sparse matrix to CSC format for scipy
    // Get the data
    py::array_t<double> data(mat.n_nonzero, mat.values);
    py::array_t<size_t> indices(mat.n_nonzero, mat.row_indices);
    py::array_t<size_t> indptr(mat.n_cols + 1, mat.col_ptrs);
    
    py::tuple shape = py::make_tuple(mat.n_rows, mat.n_cols);
    
    py::dict result;
    result["data"] = data;
    result["indices"] = indices;
    result["indptr"] = indptr;
    result["shape"] = shape;
    return result;
}

PYBIND11_MODULE(_operators, m) {
    // Gradient operator
    py::class_<Gradient, sp_mat>(m, "GradientOperator")
        .def(py::init<u16, u32, Real>())
        .def("matrix", [](const Gradient& op) {
            return sparse_to_numpy(op);
        });

    // Divergence operator
    py::class_<Divergence, sp_mat>(m, "DivergenceOperator")
        .def(py::init<u16, u32, Real>())
        .def("matrix", [](const Divergence& op) {
            return sparse_to_numpy(op);
        });

    // Laplacian operator
    py::class_<Laplacian, sp_mat>(m, "LaplacianOperator")
        .def(py::init<u16, u32, Real>())
        .def("matrix", [](const Laplacian& op) {
            return sparse_to_numpy(op);
        });
}
