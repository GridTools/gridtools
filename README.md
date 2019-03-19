<a target="_blank" href="https://opensource.org/licenses/BSD-3-Clause">![License: BSD][BSD.License]</a>

<a href="https://GridTools.github.io/gridtools"><img src="docs/_static/logo.svg"/></a>

GridTools
=========

The GridTools framework is a set of libraries and utilities to develop performance portable applications in which stencil operations on grids are central. A detailed introduction can be found in the [documentation](https://GridTools.github.io/gridtools).

Installation instructions
-------------------------

```
git clone https://github.com/GridTools/gridtools.git
cd gridtools
mkdir -p build && cd build
cmake ..
cmake --build .
```

Supported compilers
-------------------

The GridTools libraries are currently nightly tested with the following compilers on [CSCS supercomputers](https://www.cscs.ch/computers/overview/).

| Compiler | Target | Tested on |
| --- | --- | --- |
| NVCC 9.1 with GNU 5.3 | cuda | Piz Daint |
| NVCC 9.1 with Clang 3.8.1 | cuda | Piz Daint |
| GNU 7.1.0 | x86, mc | Piz Daint |
| Clang 7.0.1 | x86, mc | Piz Daint |
| NVCC 8.0 with GNU 5.4.0 | cuda | Piz Kesch |
| Intel 18.0.2 | x86, mc | Grand Tave |

Contributing
------------
Contributions to the GridTools framework are welcome. Please open an issue for any bugs that you encounter or provide a fix or enhancement as a PR. External contributions to GridTools require us a signed copy of a copyright release form to ETH Zurich. We will contact you on the PR.

[BSD.License]: https://img.shields.io/badge/License-BSD-blue.svg
