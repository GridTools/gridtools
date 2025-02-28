<a href="https://GridTools.github.io/gridtools"><img src="https://raw.githubusercontent.com/GridTools/gridtools/gh-pages/v1.0/_static/logo.svg"/></a>
<br/><br/>
<a target="_blank" href="https://opensource.org/licenses/BSD-3-Clause">![License: BSD][BSD.License]</a>
![](https://github.com/GridTools/gridtools/workflows/CI/badge.svg?branch=master)
![](https://github.com/GridTools/gridtools/workflows/CMake-config/badge.svg?branch=master)
<a target="_blank" href="https://join.slack.com/t/gridtools/shared_invite/zt-1mceuj747-59swuowC3MKAuCFyNAnc1g"><img src="https://img.shields.io/badge/slack-join-orange?logo=slack"></a>
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/GridTools/gridtools) 

The GridTools framework is a set of libraries and utilities to develop performance portable applications in the area of weather and climate. To achieve the goal of performance portability, the user-code is written in a generic form which is then optimized for a given architecture at compile-time. The core of GridTools is the stencil composition module which implements a DSL embedded in C++ for stencils and stencil-like patterns. Further, GridTools provides modules for halo exchanges, boundary conditions, data management and bindings to C and Fortran.

GridTools is successfully used to accelerate the dynamical core of the [COSMO model](http://cosmo-model.org/) with improved performance on CUDA-GPUs compared to the current official version, demonstrating production quality and feature-completeness of the library for models on lat-lon grids. The GridTools-based dynamical core is shipped with COSMO v5.7 and later, see [release notes COSMO v5.7](http://cosmo-model.org/content/model/releases/histories/cosmo_5.07.htm).

Although GridTools was developed for weather and climate applications it might be applicable for other domains with a focus on stencil-like computations.

A detailed introduction can be found in the [documentation](https://GridTools.github.io/gridtools).

### Installation instructions

```
git clone https://github.com/GridTools/gridtools.git
cd gridtools
mkdir -p build && cd build
cmake ..
make -j8
make test
```

For choosing the compiler, use the standard CMake techniques, e.g. setting the environment variables
```
CXX=`which g++` # full path to the C++ compiler
CC=`which gcc` # full path to theC compiler
FC=`which gfortran` # full path to theFortran compiler
CUDACXX=`which nvcc` # full path to NVCC
CUDAHOSTCXX=`which g++` # full path to the C++ compiler to be used as CUDA host compiler
```

##### Requirements
- C++17 compiler (see also list of tested compilers)
- CMake (3.21.0 or later)
- CUDA Toolkit (11.0 or later, optional)
- MPI (optional, CUDA-aware MPI for the GPU communication module `gcl_gpu`)

### Supported compilers

The GridTools libraries are currently nightly tested with the following compilers on [CSCS supercomputers](https://www.cscs.ch/computers/overview/).

| Compiler | Backend | Tested on | Comments |
| --- | --- | --- | --- |
| Cray clang version 12.0.3 | all backends | Piz Daint | P100 GPU | with Clang-CUDA
| Cray clang version 10.0.2 + NVCC 11.2 | all backends | Piz Daint | P100 GPU | 
| Cray clang version 12.0.3 | all backends | Piz Daint | with -std=c++20
| GNU 11.2.0 + NVCC 11.0 | all backends | Piz Daint | P100 GPU |
| GNU 11.2.0 + NVCC 11.2 | all backends | Dom | P100 GPU |
| GNU 8.3.0 + NVCC 11.2 | all backends | Tsa | V100 GPU |

##### Known issues

- CUDA 11.0.x has a severe issue, see https://github.com/GridTools/gridtools/issues/1522. Under certain conditions, GridTools code will not compile for this version of CUDA. CUDA 11.1.x and later should not be affected by this issue.
- CUDA 12.1, 12.2, 12.3, 12.4 have various issues related to `constexpr`, see https://github.com/GridTools/gridtools/issues/1766. We recommend CUDA 12.5 or later.
- Cray Clang version 11.0.0 has a problem with the `gridtools::tuple` conversion constructor, see https://github.com/GridTools/gridtools/issues/1615.

##### Partly supported (expected to work, but not tested regularly)

| Compiler | Backend | Date | Comments |
| --- | --- | --- | --- |
| Intel 19.1.1.217 | all backends | 2021-09-30 | with `cmake . -DCMAKE_CXX_FLAGS=-qnextgen` |
| NVHPC 23.3 | all backends | 2023-04-20 | only compilation is tested regularly in CI |
| ROCm 6.0.3 | all backends | 2024-09-24 | tested on AMD MI250X (LUMI) |

### Contributing

Contributions to the GridTools framework are welcome. Please open an issue for any bugs that you encounter or provide a fix or enhancement as a PR. External contributions to GridTools require us a signed copy of a [copyright release form to ETH Zurich](https://github.com/GridTools/CAA). We will contact you on the PR.

[BSD.License]: https://img.shields.io/badge/License-BSD--3--Clause-blue.svg
