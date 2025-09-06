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
grad = pymole.create_gradient(100, 0.1)

# Switch to C++ backend for performance
pymole.use_backend('cpp')
grad = pymole.create_gradient(100, 0.1)

# Apply operator
x = np.random.rand(100)
result = grad @ x
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

Based on the original MOLE library:
- Repository: https://github.com/csrc-sdsu/mole
- License: GPL-3.0
