/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/structured_grids/grid.hpp>

using namespace gridtools;

template <typename Axis>
GT_FUNCTION bool test_grid_eq(grid<Axis> &expect, grid<Axis> &actual) {
    bool result = expect.direction_i() == actual.direction_i();
    result &= expect.direction_j() == actual.direction_j();
    for (int i = 0; i < expect.value_list.size(); ++i) {
        result &= expect.value_list[i] == actual.value_list[i];
    }
    return result;
}
