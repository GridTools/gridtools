import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import py_implementation as testee

def test_3d():
    src = np.fromfunction(lambda i, j, k : i + j + k, (3, 4, 5), dtype=np.double)
    dst = np.empty_like(src)
    testee.copy_from_3D(src, dst)
    assert np.all(dst == src)

def test_1d():
    shape = (3, 4, 5)
    src = np.arange(shape[0], dtype=np.double)
    dst = np.empty(shape, dtype=np.double)
    testee.copy_from_1D(src, dst)
    expected = np.fromfunction(lambda i, j, k : i, shape, dtype=np.double)
    assert np.all(dst == expected)

def test_scalar():
    dst = np.empty((3, 4, 5), dtype=np.double)
    testee.copy_from_scalar(42, dst)
    assert np.all(dst == 42)

test_3d()
test_1d()
test_scalar()
