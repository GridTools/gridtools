/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../grid.hpp"

namespace gridtools {
    GT_FUNCTION GT_HOST_CONSTEXPR uint_t block_i_size(backend::naive const &) { return 0; }
    GT_FUNCTION GT_HOST_CONSTEXPR uint_t block_j_size(backend::naive const &) { return 0; }

    template <class Grid>
    uint_t block_i_size(backend::naive const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.i_high_bound() - grid.i_low_bound() + 1;
    }
    template <class Grid>
    uint_t block_j_size(backend::naive const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.j_high_bound() - grid.j_low_bound() + 1;
    }
} // namespace gridtools
