# Development

The `gridtools-cpp` distribution is built from the repository root with
[scikit-build-core](https://scikit-build-core.readthedocs.io/) (see the top-level
`pyproject.toml`). The build runs the root `CMakeLists.txt` to install the headers and
CMake config into the package data, and reads the version from `version.txt` — there is
no separate preparation step.

## Installing

As always it is recommended to carry out the following steps in a virtual environment.
From the repository root:

```bash
pip install .
```

Directly from a git ref (no subdirectory needed):

```bash
pip install "git+https://github.com/GridTools/gridtools.git@<ref>"
```

## Building for distribution

```bash
python -m build --wheel
```

The header-only wheel is platform independent (`gridtools_cpp-<version>-py3-none-any.whl`).

## Testing

Using nox (from this directory), the tests run in isolated environments:

```bash
nox -s test_src
```

To build and test the wheel distribution specifically:

```bash
nox -s build_wheel test_wheel_with_python-3.10  # replace 3.10 with the Python version you are running
```

To run against all supported versions (requires Python 3.8, 3.9, 3.10 and 3.11 in your path):

```bash
nox
```
