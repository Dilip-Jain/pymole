# PyMOLE

Python implementation of MOLE (Mimetic Operators Library Enhanced)

## Description

PyMOLE is a Python implementation of the SDSU's MOLE library, providing mimetic operators for numerical calculations. It offers both pure Python and C++ implementations, allowing users to choose between ease of installation and maximum performance.

## Installation

Basic installation (Python-only):
```bash
pip install pymole
```

Full installation (with C++ backend):
```bash
pip install pymole[cpp]
```

## Usage

```python
import pymole
import numpy as np

# Create a gradient operator (uses Python backend by default)
m, dx = 100, 0.01
grad = pymole.create_gradient(m, dx)

# Switch to C++ backend for performance
pymole.use_backend('cpp')
grad = pymole.create_gradient(m, dx)

# Non-periodic MOLE grids use staggered vectors: G is (m+1) x (m+2).
x = np.linspace(0.0, 1.0, m + 2)
result = grad @ x
```

For non-periodic 1D operators, gradient maps `m + 2` nodal values to `m + 1` values, divergence maps `m + 1` values to `m + 2`, and the Laplacian is `(m + 2) x (m + 2)` and is assembled as divergence times gradient. The supplied spacing is used directly.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

Based on the original MOLE library:
- Repository: https://github.com/csrc-sdsu/mole
- License: GPL-3.0
