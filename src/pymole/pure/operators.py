"""Pure Python implementation of mimetic operators."""

from typing import Literal, List, Union, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix, csc_matrix

from ..base import MimeticOperator


_MOLE_PERIODIC_GRADIENT_STENCILS = {
    2: ((1, 1.0), (2, -1.0)),
    4: ((0, -1.0 / 24.0), (1, 9.0 / 8.0), (2, -9.0 / 8.0), (3, 1.0 / 24.0)),
    6: ((0, -25.0 / 384.0), (1, 75.0 / 64.0), (2, -75.0 / 64.0),
        (3, 25.0 / 384.0), (4, -3.0 / 640.0), (-1, 3.0 / 640.0)),
    8: ((0, -245.0 / 3072.0), (1, 1225.0 / 1024.0), (2, -1225.0 / 1024.0),
        (3, 245.0 / 3072.0), (4, -49.0 / 5120.0), (5, 5.0 / 7168.0),
        (-2, -5.0 / 7168.0), (-1, 49.0 / 5120.0)),
}


def _build_mole_periodic_gradient(m: int, dx: float, k: int) -> csc_matrix:
    try:
        stencil = _MOLE_PERIODIC_GRADIENT_STENCILS[k]
    except KeyError as exc:
        raise ValueError("MOLE supports k in {2, 4, 6, 8}") from exc
    if m < 2 * k:
        raise ValueError(f"MOLE requires m >= 2*k; got m={m}, k={k}")

    matrix = np.zeros((m, m), dtype=float)
    indices = np.arange(m)
    for offset, coefficient in stencil:
        matrix[indices, (indices - offset) % m] = coefficient / dx
    return csc_matrix(matrix)


def _set_mole_rows(matrix: np.ndarray, rows: List[Tuple[int, int, Tuple[float, ...]]]) -> None:
    for row, start, coefficients in rows:
        matrix[row, np.arange(start, start + len(coefficients))] = coefficients


_MOLE_GRADIENT_NONPERIODIC = {
    2: (
        (1, 0, ((-1.0, 1.0),)),
        ((0, 0, (-8.0 / 3.0, 3.0, -1.0 / 3.0)),),
        ((0, -1, (1.0 / 3.0, -3.0, 8.0 / 3.0)),),
    ),
    4: (
        (2, -1, ((1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0),)),
        ((0, 0, (-352.0 / 105.0, 35.0 / 8.0, -35.0 / 24.0, 21.0 / 40.0, -5.0 / 56.0)),
         (1, 0, (16.0 / 105.0, -31.0 / 24.0, 29.0 / 24.0, -3.0 / 40.0, 1.0 / 168.0))),
        ((0, -3, (5.0 / 56.0, -21.0 / 40.0, 35.0 / 24.0, -35.0 / 8.0, 352.0 / 105.0)),
         (-1, -3, (-1.0 / 168.0, 3.0 / 40.0, -29.0 / 24.0, 31.0 / 24.0, -16.0 / 105.0))),
    ),
    6: (
        (3, -2, ((-3.0 / 640.0, 25.0 / 384.0, -75.0 / 64.0,
                  75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0),)),
        ((0, 0, (-13016.0 / 3465.0, 693.0 / 128.0, -385.0 / 128.0, 693.0 / 320.0,
                -495.0 / 448.0, 385.0 / 1152.0, -63.0 / 1408.0)),
         (1, 0, (496.0 / 3465.0, -811.0 / 640.0, 449.0 / 384.0, -29.0 / 960.0,
                -11.0 / 448.0, 13.0 / 1152.0, -37.0 / 21120.0)),
         (2, 0, (-8.0 / 385.0, 179.0 / 1920.0, -153.0 / 128.0, 381.0 / 320.0,
                -101.0 / 1344.0, 1.0 / 128.0, -3.0 / 7040.0))),
        ((0, -5, (63.0 / 1408.0, -385.0 / 1152.0, 495.0 / 448.0, -693.0 / 320.0,
                  385.0 / 128.0, -693.0 / 128.0, 13016.0 / 3465.0)),
         (-1, -5, (37.0 / 21120.0, -13.0 / 1152.0, 11.0 / 448.0, 29.0 / 960.0,
                   -449.0 / 384.0, 811.0 / 640.0, -496.0 / 3465.0)),
         (-2, -5, (3.0 / 7040.0, -1.0 / 128.0, 101.0 / 1344.0, -381.0 / 320.0,
                   153.0 / 128.0, -179.0 / 1920.0, 8.0 / 385.0))),
    ),
    8: (
        (4, -3, ((5.0 / 7168.0, -49.0 / 5120.0, 245.0 / 3072.0, -1225.0 / 1024.0,
                  1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0),)),
        ((0, 0, (-4856215.0 / 1200963.0, 45858154.0 / 7297397.0, -23409299.0 / 4789435.0,
                3799178.0 / 719717.0, -4892189.0 / 1089890.0, 1789111.0 / 658879.0,
                -1406819.0 / 1289899.0, 1154863.0 / 4436807.0, -2936602.0 / 105142673.0)),
         (1, 0, (86048.0 / 675675.0, -131093.0 / 107520.0, 5503131.0 / 5166017.0,
                305249.0 / 2136437.0, -1763845.0 / 8250973.0, 1562032.0 / 10745723.0,
                -270419.0 / 4422611.0, 2983.0 / 199680.0, -2621.0 / 1612800.0)),
         (2, 0, (-3776.0 / 225225.0, 8707.0 / 107520.0, -17947.0 / 15360.0,
                29319.0 / 25600.0, -533.0 / 21504.0, -263.0 / 9216.0,
                903.0 / 56320.0, -283.0 / 66560.0, 257.0 / 537600.0)),
         (3, 0, (32.0 / 9009.0, -543.0 / 35840.0, 265.0 / 3072.0, -1233.0 / 1024.0,
                8625.0 / 7168.0, -775.0 / 9216.0, 639.0 / 56320.0,
                -15.0 / 13312.0, 1.0 / 21504.0))),
        ((0, -7, (2936602.0 / 105142673.0, -1154863.0 / 4436807.0, 1406819.0 / 1289899.0,
                  -1789111.0 / 658879.0, 4892189.0 / 1089890.0, -3799178.0 / 719717.0,
                  23409299.0 / 4789435.0, -45858154.0 / 7297397.0, 4856215.0 / 1200963.0)),
         (-1, -7, (2621.0 / 1612800.0, -2983.0 / 199680.0, 270419.0 / 4422611.0,
                   -1562032.0 / 10745723.0, 1763845.0 / 8250973.0, -305249.0 / 2136437.0,
                   -5503131.0 / 5166017.0, 131093.0 / 107520.0, -86048.0 / 675675.0)),
         (-2, -7, (-257.0 / 537600.0, 283.0 / 66560.0, -903.0 / 56320.0, 263.0 / 9216.0,
                   533.0 / 21504.0, -29319.0 / 25600.0, 17947.0 / 15360.0,
                   -8707.0 / 107520.0, 3776.0 / 225225.0)),
         (-3, -7, (-1.0 / 21504.0, 15.0 / 13312.0, -639.0 / 56320.0, 775.0 / 9216.0,
                   -8625.0 / 7168.0, 1233.0 / 1024.0, -265.0 / 3072.0,
                   543.0 / 35840.0, -32.0 / 9009.0))),
    ),
}


def _build_mole_nonperiodic(m: int, dx: float, k: int, divergence: bool) -> csc_matrix:
    if k not in (2, 4, 6, 8):
        raise ValueError("MOLE supports k in {2, 4, 6, 8}")
    if divergence and m <= 2 * k:
        raise ValueError(f"MOLE requires m > 2*k; got m={m}, k={k}")
    if not divergence and m < 2 * k:
        raise ValueError(f"MOLE requires m >= 2*k; got m={m}, k={k}")

    if divergence:
        shape = (m + 2, m + 1)
        interior_start, interior_stop, interior_cols, interior_values = {
            2: (1, m + 1, (-1, 0), (-1.0, 1.0)),
            4: (2, m, (-2, -1, 0, 1), (1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0)),
            6: (3, m - 1, (-3, -2, -1, 0, 1, 2),
                (-3.0 / 640.0, 25.0 / 384.0, -75.0 / 64.0, 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0)),
            8: (4, m - 2, (-4, -3, -2, -1, 0, 1, 2, 3),
                (5.0 / 7168.0, -49.0 / 5120.0, 245.0 / 3072.0, -1225.0 / 1024.0,
                 1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0)),
        }[k]
        left_rows, right_rows = _MOLE_DIVERGENCE_BOUNDARIES[k]
    else:
        shape = (m + 1, m + 2)
        interior_start, interior_stop, interior_cols, interior_values = {
            2: (1, m, (0, 1), (-1.0, 1.0)),
            4: (2, m - 1, (-1, 0, 1, 2), (1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0)),
            6: (3, m - 2, (-2, -1, 0, 1, 2, 3),
                (-3.0 / 640.0, 25.0 / 384.0, -75.0 / 64.0, 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0)),
            8: (4, m - 3, (-3, -2, -1, 0, 1, 2, 3, 4),
                (5.0 / 7168.0, -49.0 / 5120.0, 245.0 / 3072.0, -1225.0 / 1024.0,
                 1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0)),
        }[k]
        left_rows, right_rows = _MOLE_GRADIENT_NONPERIODIC[k][1:]

    matrix = np.zeros(shape, dtype=float)
    for row in range(interior_start, interior_stop):
        matrix[row, row + np.asarray(interior_cols)] = interior_values
    _set_mole_rows(matrix, [(row, start, values) for row, start, values in left_rows])
    _set_mole_rows(matrix, [(m + row, m + start, values) for row, start, values in right_rows])
    return csc_matrix(matrix / dx)


def _normalize_grid(n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]]):
    dimensions = (n,) if isinstance(n, int) else tuple(n)
    spacings = (h,) if isinstance(h, (int, float)) else tuple(h)
    if not 1 <= len(dimensions) <= 3 or len(dimensions) != len(spacings):
        raise ValueError("Grid dimensions and spacings must have matching 1D, 2D, or 3D lengths")
    if any(size <= 0 for size in dimensions) or any(step <= 0 for step in spacings):
        raise ValueError("Grid dimensions and spacings must be positive")
    return dimensions, spacings


def _trimmed_identity_rows(size: int) -> csc_matrix:
    return csc_matrix(sparse.eye(size + 2, format="csc")[1:-1, :])


def _trimmed_identity_cols(size: int) -> csc_matrix:
    return csc_matrix(sparse.eye(size + 2, format="csc")[:, 1:-1])


def _axis_gradient(size: int, spacing: float, k: int, boundary: str) -> csc_matrix:
    if boundary == "periodic":
        return _build_mole_periodic_gradient(size, spacing, k)
    if boundary != "nonperiodic":
        raise ValueError("boundary must be 'periodic' or 'nonperiodic'")
    return _build_mole_nonperiodic(size, spacing, k, divergence=False)


def _axis_divergence(size: int, spacing: float, k: int, boundary: str) -> csc_matrix:
    if boundary == "periodic":
        return (-_build_mole_periodic_gradient(size, spacing, k).T).tocsc()
    if boundary != "nonperiodic":
        raise ValueError("boundary must be 'periodic' or 'nonperiodic'")
    return _build_mole_nonperiodic(size, spacing, k, divergence=True)


def _build_mole_gradient(dimensions: Tuple[int, ...], spacings: Tuple[float, ...],
                         k: int, boundary: str) -> csc_matrix:
    axes = [_axis_gradient(size, step, k, boundary) for size, step in zip(dimensions, spacings)]
    selectors = [
        csc_matrix(sparse.eye(size, format="csc")) if boundary == "periodic"
        else _trimmed_identity_rows(size)
        for size in dimensions
    ]
    if len(dimensions) == 1:
        return axes[0]
    if len(dimensions) == 2:
        gx = sparse.kron(selectors[1], axes[0], format="csc")
        gy = sparse.kron(axes[1], selectors[0], format="csc")
        return csc_matrix(sparse.vstack((gx, gy), format="csc"))

    gx = sparse.kron(sparse.kron(selectors[2], selectors[1], format="csc"), axes[0], format="csc")
    gy = sparse.kron(sparse.kron(selectors[2], axes[1], format="csc"), selectors[0], format="csc")
    gz = sparse.kron(sparse.kron(axes[2], selectors[1], format="csc"), selectors[0], format="csc")
    return csc_matrix(sparse.vstack((gx, gy, gz), format="csc"))


def _build_mole_divergence(dimensions: Tuple[int, ...], spacings: Tuple[float, ...],
                           k: int, boundary: str) -> csc_matrix:
    axes = [_axis_divergence(size, step, k, boundary) for size, step in zip(dimensions, spacings)]
    selectors = [
        csc_matrix(sparse.eye(size, format="csc")) if boundary == "periodic"
        else _trimmed_identity_cols(size)
        for size in dimensions
    ]
    if len(dimensions) == 1:
        return axes[0]
    if len(dimensions) == 2:
        dx = sparse.kron(selectors[1], axes[0], format="csc")
        dy = sparse.kron(axes[1], selectors[0], format="csc")
        return csc_matrix(sparse.hstack((dx, dy), format="csc"))

    dx = sparse.kron(sparse.kron(selectors[2], selectors[1], format="csc"), axes[0], format="csc")
    dy = sparse.kron(sparse.kron(selectors[2], axes[1], format="csc"), selectors[0], format="csc")
    dz = sparse.kron(sparse.kron(axes[2], selectors[1], format="csc"), selectors[0], format="csc")
    return csc_matrix(sparse.hstack((dx, dy, dz), format="csc"))


_MOLE_DIVERGENCE_BOUNDARIES = {
    2: ((), ()),
    4: (
    ((1, 0, (-11.0 / 12.0, 17.0 / 24.0, 3.0 / 8.0, -5.0 / 24.0, 1.0 / 24.0)),),
    ((0, -4, (-1.0 / 24.0, 5.0 / 24.0, -3.0 / 8.0, -17.0 / 24.0, 11.0 / 12.0)),),
    ),
    6: (
    ((1, 0, (-1627.0 / 1920.0, 211.0 / 640.0, 59.0 / 48.0, -235.0 / 192.0,
         91.0 / 128.0, -443.0 / 1920.0, 31.0 / 960.0)),
     (2, 0, (31.0 / 960.0, -687.0 / 640.0, 129.0 / 128.0, 19.0 / 192.0,
         -3.0 / 32.0, 21.0 / 640.0, -3.0 / 640.0))),
    ((0, -6, (-31.0 / 960.0, 443.0 / 1920.0, -91.0 / 128.0, 235.0 / 192.0,
          -59.0 / 48.0, -211.0 / 640.0, 1627.0 / 1920.0)),
     (-1, -6, (3.0 / 640.0, -21.0 / 640.0, 3.0 / 32.0, -19.0 / 192.0,
           -129.0 / 128.0, 687.0 / 640.0, -31.0 / 960.0))),
    ),
    8: (
    ((1, 0, (-1423.0 / 1792.0, -491.0 / 7168.0, 7753.0 / 3072.0, -18509.0 / 5120.0,
         3535.0 / 1024.0, -2279.0 / 1024.0, 953.0 / 1024.0,
         -1637.0 / 7168.0, 2689.0 / 107520.0)),
     (2, 0, (2689.0 / 107520.0, -36527.0 / 35840.0, 4259.0 / 5120.0, 6497.0 / 15360.0,
         -475.0 / 1024.0, 1541.0 / 5120.0, -639.0 / 5120.0,
         1087.0 / 35840.0, -59.0 / 17920.0)),
     (3, 0, (-59.0 / 17920.0, 1175.0 / 21504.0, -1165.0 / 1024.0, 1135.0 / 1024.0,
         25.0 / 3072.0, -251.0 / 5120.0, 25.0 / 1024.0,
         -45.0 / 7168.0, 5.0 / 7168.0))),
    ((0, -8, (-2689.0 / 107520.0, 1637.0 / 7168.0, -953.0 / 1024.0, 2279.0 / 1024.0,
          -3535.0 / 1024.0, 18509.0 / 5120.0, -7753.0 / 3072.0,
          491.0 / 7168.0, 1423.0 / 1792.0)),
     (-1, -8, (59.0 / 17920.0, -1087.0 / 35840.0, 639.0 / 5120.0, -1541.0 / 5120.0,
           475.0 / 1024.0, -6497.0 / 15360.0, -4259.0 / 5120.0,
           36527.0 / 35840.0, -2689.0 / 107520.0)),
     (-2, -8, (-5.0 / 7168.0, 45.0 / 7168.0, -25.0 / 1024.0, 251.0 / 5120.0,
           -25.0 / 3072.0, -1135.0 / 1024.0, 1165.0 / 1024.0,
           -1175.0 / 21504.0, 59.0 / 17920.0))),
    ),
}


class _PureMimeticOperator(MimeticOperator):
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self @ x


class MimeticGradient(_PureMimeticOperator):
    """MOLE 1D gradient, including its periodic and staggered non-periodic grids."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic',
                 k: int = 2):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.boundary = boundary
        self.k = k
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> sparse.spmatrix:
        return _build_mole_gradient(self._dimensions, self._spacings, self.k, self.boundary)


class MimeticDivergence(_PureMimeticOperator):
    """MOLE 1D divergence, including its periodic and staggered grids."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic',
                 k: int = 2):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.boundary = boundary
        self.k = k
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> sparse.spmatrix:
        return _build_mole_divergence(self._dimensions, self._spacings, self.k, self.boundary)


class MimeticLaplacian(_PureMimeticOperator):
    """MOLE 1D Laplacian, assembled from divergence times gradient."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic',
                 k: int = 2):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.boundary = boundary
        self.k = k
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> spmatrix:
        divergence = MimeticDivergence(self._dimensions, self._spacings, self.boundary, k=self.k)
        gradient = MimeticGradient(self._dimensions, self._spacings, self.boundary, k=self.k)
        return (divergence.matrix @ gradient.matrix).tocsc()


class MimeticInterpol(_PureMimeticOperator):
    """MOLE interpolation from nodal values to staggered values."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 c: Union[float, Tuple[float, ...]] = 0.5,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        weights = (c,) * len(dimensions) if isinstance(c, (int, float)) else tuple(c)
        if len(weights) != len(dimensions) or any(not 0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError("Interpolation weights must match the dimension and be in [0, 1]")
        if any(size < 4 for size in dimensions):
            raise ValueError("MOLE interpolation requires at least 4 cells per dimension")
        if boundary != 'nonperiodic':
            raise ValueError("MOLE interpolation supports only non-periodic boundaries")
        self._weights = weights
        self.c = c
        self.boundary = boundary
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> spmatrix:
        matrices = []
        for size, weight in zip(self._dimensions, self._weights):
            matrix = np.zeros((size + 1, size + 2), dtype=float)
            matrix[0, 0] = 1.0
            matrix[size, size + 1] = 1.0
            rows = np.arange(1, size)
            matrix[rows, rows] = weight
            matrix[rows, rows + 1] = 1.0 - weight
            matrices.append(csc_matrix(matrix))

        selectors = [_trimmed_identity_rows(size) for size in self._dimensions]
        if len(self._dimensions) == 1:
            return matrices[0]
        if len(self._dimensions) == 2:
            mx = sparse.kron(selectors[1], matrices[0], format="csc")
            my = sparse.kron(matrices[1], selectors[0], format="csc")
            return csc_matrix(sparse.vstack((mx, my), format="csc"))

        mx = sparse.kron(sparse.kron(selectors[2], selectors[1], format="csc"), matrices[0], format="csc")
        my = sparse.kron(sparse.kron(selectors[2], matrices[1], format="csc"), selectors[0], format="csc")
        mz = sparse.kron(sparse.kron(matrices[2], selectors[1], format="csc"), selectors[0], format="csc")
        return csc_matrix(sparse.vstack((mx, my, mz), format="csc"))


def _build_mole_robin_axis(m: int, dx: float, k: int, a: float, b: float) -> csc_matrix:
    gradient = _build_mole_nonperiodic(m, dx, k, divergence=False)
    matrix = sparse.lil_matrix((m + 2, m + 2), dtype=float)
    matrix[0, 0] = a
    matrix[m + 1, m + 1] = a
    matrix[0, :] -= b * gradient.getrow(0)
    matrix[m + 1, :] += b * gradient.getrow(m)
    return matrix.tocsc()


def _boundary_mask(size: int) -> csc_matrix:
    mask = sparse.eye(size + 2, format="lil")
    mask[0, 0] = 0.0
    mask[size + 1, size + 1] = 0.0
    return mask.tocsc()


def _assemble_mole_boundary(dims: Tuple[int, ...], axes: List[csc_matrix]) -> csc_matrix:
    identities = [csc_matrix(sparse.eye(size + 2, format="csc")) for size in dims]
    if len(dims) == 1:
        return axes[0]
    if len(dims) == 2:
        mask_y = _boundary_mask(dims[1])
        return csc_matrix(
            sparse.kron(mask_y, axes[0], format="csc")
            + sparse.kron(axes[1], identities[0], format="csc")
        )

    mask_y = _boundary_mask(dims[1])
    mask_z = _boundary_mask(dims[2])
    return csc_matrix(
        sparse.kron(sparse.kron(mask_z, mask_y, format="csc"), axes[0], format="csc")
        + sparse.kron(sparse.kron(mask_z, axes[1], format="csc"), identities[0], format="csc")
        + sparse.kron(sparse.kron(axes[2], identities[1], format="csc"), identities[0], format="csc")
    )


class MimeticRobinBC(_PureMimeticOperator):
    """MOLE Robin boundary-condition matrix for 1D, 2D, or 3D grids."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 k: int = 2, a: float = 1.0, b: float = 0.0):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.k = k
        self.a = a
        self.b = b
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> csc_matrix:
        axes = [
            _build_mole_robin_axis(size, step, self.k, self.a, self.b)
            for size, step in zip(self._dimensions, self._spacings)
        ]
        return _assemble_mole_boundary(self._dimensions, axes)


def _build_mole_mixed_axis(m: int, dx: float, k: int, left: str,
                           coeffs_left: List[float], right: str,
                           coeffs_right: List[float]) -> csc_matrix:
    gradient = _build_mole_nonperiodic(m, dx, k, divergence=False)
    matrix = sparse.lil_matrix((m + 2, m + 2), dtype=float)

    for side, condition, coefficients in (
        (0, left, coeffs_left),
        (m + 1, right, coeffs_right),
    ):
        if condition not in ("Dirichlet", "Neumann", "Robin"):
            raise ValueError(f"Unknown boundary condition type: {condition}")
        required = 2 if condition == "Robin" else 1
        if len(coefficients) < required:
            raise ValueError(f"{condition} boundary requires {required} coefficient(s)")
        if condition in ("Dirichlet", "Robin"):
            matrix[side, side] += coefficients[0]
        if condition == "Neumann":
            derivative = coefficients[0]
        elif condition == "Robin":
            derivative = coefficients[1]
        else:
            continue
        gradient_row = 0 if side == 0 else m
        sign = -1.0 if side == 0 else 1.0
        matrix[side, :] += sign * derivative * gradient.getrow(gradient_row)

    return matrix.tocsc()


class MimeticMixedBC(_PureMimeticOperator):
    """MOLE mixed Dirichlet, Neumann, and Robin boundary matrix."""

    def __init__(self, n: Union[int, Tuple[int, ...]], h: Union[float, Tuple[float, ...]],
                 k: int = 2, **bc_dict):
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self._dimensions = dimensions
        self._spacings = spacings
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.k = k
        self._bc_dict = bc_dict
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> csc_matrix:
        names = (("left", "right"), ("bottom", "top"), ("front", "back"))
        axes = []
        for axis, (size, step) in enumerate(zip(self._dimensions, self._spacings)):
            left_name, right_name = names[axis]
            try:
                left = self._bc_dict[left_name]
                right = self._bc_dict[right_name]
                coeffs_left = self._bc_dict[f"coeffs_{left_name}"]
                coeffs_right = self._bc_dict[f"coeffs_{right_name}"]
            except KeyError as exc:
                raise ValueError(f"Missing mixed boundary argument: {exc.args[0]}") from exc
            axes.append(
                _build_mole_mixed_axis(size, step, self.k, left, coeffs_left, right, coeffs_right)
            )
        return _assemble_mole_boundary(self._dimensions, axes)

