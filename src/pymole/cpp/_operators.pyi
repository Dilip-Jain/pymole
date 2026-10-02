"""Type declarations for the compiled MOLE operator module."""

# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-arguments,too-many-positional-arguments,too-many-locals,unused-argument

from typing import Sequence, Tuple, overload

from numpy import float64
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

_Vector = NDArray[float64]


class Gradient:
    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(self, k: int, m: int, n: int, dx: float, dy: float) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...


class Divergence:
    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(self, k: int, m: int, n: int, dx: float, dy: float) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...


class Laplacian:
    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(self, k: int, m: int, n: int, dx: float, dy: float) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...


class Interpol:
    @overload
    def __init__(self, m: int, c: float) -> None: ...
    @overload
    def __init__(self, m: int, n: int, c1: float, c2: float) -> None: ...
    @overload
    def __init__(
        self,
        m: int,
        n: int,
        o: int,
        c1: float,
        c2: float,
        c3: float,
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...


class RobinBC:
    @overload
    def __init__(self, k: int, m: int, dx: float, a: float, b: float) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        a: float,
        b: float,
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        o: int,
        dz: float,
        a: float,
        b: float,
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...


class MixedBC:
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
        bottom: str,
        coeffs_bottom: Sequence[float],
        top: str,
        coeffs_top: Sequence[float],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        o: int,
        dz: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
        bottom: str,
        coeffs_bottom: Sequence[float],
        top: str,
        coeffs_top: Sequence[float],
        front: str,
        coeffs_front: Sequence[float],
        back: str,
        coeffs_back: Sequence[float],
    ) -> None: ...

    def to_scipy_sparse(self) -> csc_matrix: ...
    def apply(self, v: _Vector) -> _Vector: ...
    def __matmul__(self, v: _Vector) -> _Vector: ...
    def shape(self) -> Tuple[int, int]: ...
    def nnz(self) -> int: ...
