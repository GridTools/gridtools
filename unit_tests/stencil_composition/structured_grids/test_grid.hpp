/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/structured_grids/grid.hpp>

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
