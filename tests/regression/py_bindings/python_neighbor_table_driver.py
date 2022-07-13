import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import python_neighbor_table_implementation as testee

def test_neighbor_table():
    tbl = np.asarray([[1,2,3],[4,5,6]], dtype=np.int32)
    print(tbl.shape)
    print(testee.get_neighbor_0(tbl, 0))
    print(testee.get_neighbor_1(tbl, 0))
    print(testee.get_neighbor_2(tbl, 0))
    print(testee.get_neighbor_0(tbl, 1))
    print(testee.get_neighbor_1(tbl, 1))
    print(testee.get_neighbor_2(tbl, 1))

test_neighbor_table()
