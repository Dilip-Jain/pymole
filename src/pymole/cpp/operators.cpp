#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#define ARMA_USE_OPENMP
#define ARMA_USE_SUPERLU
#include <armadillo>

#include <gradient.h>
#include <divergence.h>
#include <laplacian.h>
#include <interpol.h>
#include <robinbc.h>
#include <mixedbc.h>

namespace py = pybind11;
using namespace arma;

/**
 * Convert Armadillo sparse matrix to scipy-compatible format (CSC)
 * Returns a dictionary with 'data', 'indices', 'indptr', 'shape' keys
 */
template<typename T>
py::object sparse_to_scipy(const T& mat) {
    // Convert to scipy sparse format (CSC - Compressed Sparse Column)
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    
    // Get CSC matrix data from Armadillo
    const sp_mat& sp = static_cast<const sp_mat&>(mat);
    
    // Extract data, indices, and indptr
    py::array_t<double> data(sp.n_nonzero);
    py::array_t<arma::uword> indices(sp.n_nonzero);
    py::array_t<arma::uword> indptr(sp.n_cols + 1);
    
    auto data_ptr = static_cast<double*>(data.request().ptr);
    auto indices_ptr = static_cast<arma::uword*>(indices.request().ptr);
    auto indptr_ptr = static_cast<arma::uword*>(indptr.request().ptr);
    
    // Copy data from Armadillo sparse matrix
    std::copy(sp.values, sp.values + sp.n_nonzero, data_ptr);
    std::copy(sp.row_indices, sp.row_indices + sp.n_nonzero, indices_ptr);
    std::copy(sp.col_ptrs, sp.col_ptrs + sp.n_cols + 1, indptr_ptr);
    
    // Create scipy.sparse.csc_matrix
    auto csc_matrix_type = scipy_sparse.attr("csc_matrix");
    auto result = csc_matrix_type(
        py::make_tuple(data, indices, indptr),
        py::make_tuple((int)sp.n_rows, (int)sp.n_cols)
    );
    
    return result;
}

/**
 * Apply operator to a numpy array (dense vector)
 */
template<typename T>
py::array_t<double> apply_operator(const T& op, py::array_t<double> v) {
    auto buf = v.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<ssize_t> shape = buf.shape;
    
    // Convert numpy array to Armadillo vector
    arma::vec arma_v(ptr, shape[0], false);
    
    // Apply operator (matrix-vector multiplication)
    arma::vec result = static_cast<const sp_mat&>(op) * arma_v;
    
    // Return result as numpy array
    return py::array_t<double>(
        result.n_elem,
        result.memptr()
    );
}

PYBIND11_MODULE(_operators, m) {
    m.doc() = "PyMOLE C++ backend - Mimetic operators library";
    
    // ==================== Gradient Operator ====================
    py::class_<Gradient, sp_mat>(m, "Gradient")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Gradient operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Gradient operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Gradient>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Gradient>,
             py::arg("v"),
             "Apply gradient operator to a vector: result = G @ v")
        
        .def("__matmul__", &apply_operator<Gradient>,
             "Operator overloading: G @ v")
        
        .def("shape", [](const Gradient& g) {
            const sp_mat& sp = static_cast<const sp_mat&>(g);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Gradient& g) {
            const sp_mat& sp = static_cast<const sp_mat&>(g);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== Divergence Operator ====================
    py::class_<Divergence, sp_mat>(m, "Divergence")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Divergence operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Divergence operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Divergence>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Divergence>,
             py::arg("v"),
             "Apply divergence operator to a vector: result = D @ v")
        
        .def("__matmul__", &apply_operator<Divergence>,
             "Operator overloading: D @ v")
        
        .def("shape", [](const Divergence& d) {
            const sp_mat& sp = static_cast<const sp_mat&>(d);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Divergence& d) {
            const sp_mat& sp = static_cast<const sp_mat&>(d);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== Laplacian Operator ====================
    py::class_<Laplacian, sp_mat>(m, "Laplacian")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "3D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  dz: Grid spacing in z-direction")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Laplacian>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Laplacian>,
             py::arg("v"),
             "Apply Laplacian operator to a vector: result = L @ v")
        
        .def("__matmul__", &apply_operator<Laplacian>,
             "Operator overloading: L @ v")
        
        .def("shape", [](const Laplacian& l) {
            const sp_mat& sp = static_cast<const sp_mat&>(l);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Laplacian& l) {
            const sp_mat& sp = static_cast<const sp_mat&>(l);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== Interpolation Operator ====================
    py::class_<Interpol, sp_mat>(m, "Interpol")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Interpolation operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Interpolation operator constructor")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Interpol>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Interpol>,
             py::arg("v"),
             "Apply interpolation operator to a vector")
        
        .def("__matmul__", &apply_operator<Interpol>,
             "Operator overloading: I @ v")
        
        .def("shape", [](const Interpol& i) {
            const sp_mat& sp = static_cast<const sp_mat&>(i);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix")
        
        .def("nnz", [](const Interpol& i) {
            const sp_mat& sp = static_cast<const sp_mat&>(i);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
}
