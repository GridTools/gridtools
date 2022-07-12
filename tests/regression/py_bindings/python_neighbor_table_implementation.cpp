/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// Keep first to test for missing includes
#include <gridtools/fn/python_neighbor_table_adapter.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace gridtools;
using namespace fn;

template <class Tbl>
auto get_neighbors(Tbl const &tbl, int index) {
    return neighbor_table::neighbors(tbl, index);
}

// The module exports several instantiations of the generic `copy` to python.
// The differences between exported functions are in the way how parameters model the SID concept.
// Note that the generic algorithm stays the same.
PYBIND11_MODULE(python_neighbor_table_implementation, m) {
    m.def(
        "get_neighbor_0",
        [](py::buffer buf, int index) {
            return tuple_util::get<0>(get_neighbors(as_neighbor_table<int, 4>(buf), index));
        },
        "Return neighbors.");
    m.def(
        "get_neighbor_1",
        [](py::buffer buf, int index) {
            return tuple_util::get<1>(get_neighbors(as_neighbor_table<int, 4>(buf), index));
        },
        "Return neighbors.");
}
