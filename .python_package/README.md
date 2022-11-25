# Python package for GridTools headers and CMake files

## Usage

Use the following when compiling C++ code agains GridTools programatically from Python.
Either by calling a compiler directly or by generating a CMake project and calling CMake on it.

```python
import gridtools_cpp
include_dir = gridtools.get_include_dir()   # header files can be found here
cmake_dir = gridtools.get_cmake_dir()       # cmake files can be found here
```

## Development

In order to be able to work on this package, it is necessary to run a preparation step.
This will generate the `setup.cfg` file from `setup.cfg.in` and install the gridtools header distribution into the package data.
It will read the version number from the top-level `version.txt` and copy the `LICENSE` file to where packaging tools can find it.

All of this requires `nox`, the preparation step runs in an isolated environment and installs additional requirements `cmake` and `ninja` at runtime.

```bash
# pip install nox
# nox -s prepare
```

To delete all generated files run

```bash
# nox -s clean clean_cache
```

where `clean_cache` deletes chached files from Nox sessions like CMake builds and testing wheels (found in `.nox/.cache`), and `clean` deletes visible artifacts like `dist/`, `build/`, `.egg-info/`.
`setup.cfg` will not be deleted for convenience, to make sure tools keep functioning as expected.

### Installing

As always it is recommended to carry out the following steps in a virtual environment:

```bash
# nox -s build -- --wheel .
# pip install dist/gridtools_cpp-2.2.0-py3-none-any.whl
```

### Testing

Using nox, the tests will be carried out in isolated python environments for you:

```bash
# nox -s test_src
```

To test the wheel distribution specifically:

```bash
# nox -s build_wheel test_wheel_with_python-3.10  # replace 3.10 with the Python version you are running
```

### Advanced testing (all supported versions)

The following requires you to have Python interpreters for Python 3.8, 3.9, 3.10 and 3.11 in your system path.

```bash
# nox
```

### Building for distribution

Uses (`build`)[https://pypa-build.readthedocs.io/en/latest/], follow the link for available options.

```bash
# nox -s build -- <build options>
```
