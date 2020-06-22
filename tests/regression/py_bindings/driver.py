import os
import sys

sys.path.append(os.getcwd())

import numpy
import py_implementation

def test_3d():
    src = numpy.fromfunction(lambda i, j, k : i + j + k, (3, 4, 5), dtype=numpy.double)
    dst = numpy.empty_like(src)
    py_implementation.copy_from_3D(src, dst)
    assert numpy.all(dst == src)

def test_1d():
    shape = (3, 4, 5)
    src = numpy.arange(shape[0], dtype=numpy.double)
    dst = numpy.empty(shape, dtype=numpy.double)
    py_implementation.copy_from_1D(src, dst)
    expected = numpy.fromfunction(lambda i, j, k : i, shape, dtype=numpy.double)
    assert numpy.all(dst == expected)

def test_scalar():
    dst = numpy.empty((3, 4, 5), dtype=numpy.double)
    py_implementation.copy_from_scalar(42, dst)
    assert numpy.all(dst == 42)

test_3d()
test_1d()
test_scalar()
