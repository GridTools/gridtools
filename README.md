<a href="https://GridTools.github.io/gridtools"><img src="docs/_static/logo.svg"/></a>
<br/><br/>
<a target="_blank" href="https://opensource.org/licenses/BSD-3-Clause">![License: BSD][BSD.License]</a>

The GridTools framework is a set of libraries and utilities to develop performance portable applications in which stencil operations on grids are central. A detailed introduction can be found in the [documentation](https://GridTools.github.io/gridtools).

### Installation instructions

```
git clone https://github.com/GridTools/gridtools.git
cd gridtools
mkdir -p build && cd build
cmake ..
make -j8
make test
```
##### Requirements

- Boost (1.65.1 or later)
- CMake (3.12.4 or later)
- CUDA Toolkit (8.0 or later, optional)

### Supported compilers

The GridTools libraries are currently nightly tested with the following compilers on [CSCS supercomputers](https://www.cscs.ch/computers/overview/).

| Compiler | Backend | Tested on |
| --- | --- | --- |
| NVCC 9.1 with GNU 5.3 | cuda | Piz Daint |
| NVCC 9.1 with Clang 3.8.1 | cuda | Piz Daint |
| GNU 7.1.0 | x86, mc | Piz Daint |
| Clang 7.0.1 | x86, mc | Piz Daint |
| NVCC 8.0 with GNU 5.4.0 | cuda | Piz Kesch |

##### Known issues

- NVCC 9.2 and later have a severe performance regression. It is recommended to use NVCC <= 9.1.
- Intel is able to compile GridTools code, but depending on user code, might have severe performance problems compared to GNU- or Clang-compiled code.

##### Officially not supported (no workarounds implemented and planned)

| Compiler | Backend | Date | Comments
| --- | --- | --- | --- |
| NVCC <= 9.1 with GNU 6.x | cuda | 2018-10-16 | similar to [this tuple bug](https://devtalk.nvidia.com/default/topic/1028112/cuda-setup-and-installation/nvcc-bug-related-to-gcc-6-lt-tuple-gt-header-/)
| PGI 18.5 | x86 | 2018-12-06 | no effort to fix compilation
| Cray 8.7.3 | x86 | 2018-12-06 | no effort to fix compilation

### Contributing

Contributions to the GridTools framework are welcome. Please open an issue for any bugs that you encounter or provide a fix or enhancement as a PR. External contributions to GridTools require us a signed copy of a copyright release form to ETH Zurich. We will contact you on the PR.

[BSD.License]: https://img.shields.io/badge/License-BSD--3--Clause-blue.svg
