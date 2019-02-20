/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../backend_ids.hpp"
#include "../grid.hpp"

namespace gridtools {
    template <class GridType>
    GT_FUNCTION constexpr uint_t block_i_size(backend_ids<target::x86, GridType, strategy::block> const &) {
        return GT_DEFAULT_TILE_I;
    }
    template <class GridType>
    GT_FUNCTION constexpr uint_t block_j_size(backend_ids<target::x86, GridType, strategy::block> const &) {
        return GT_DEFAULT_TILE_J;
    }

    template <class GridType>
    GT_FUNCTION constexpr uint_t block_i_size(backend_ids<target::x86, GridType, strategy::naive> const &) {
        return 0;
    }
    template <class GridType>
    GT_FUNCTION constexpr uint_t block_j_size(backend_ids<target::x86, GridType, strategy::naive> const &) {
        return 0;
    }

    template <class GridType, class Grid>
    uint_t block_i_size(backend_ids<target::x86, GridType, strategy::naive> const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.i_high_bound() - grid.i_low_bound() + 1;
    }
    template <class GridType, class Grid>
    uint_t block_j_size(backend_ids<target::x86, GridType, strategy::naive> const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.j_high_bound() - grid.j_low_bound() + 1;
    }
} // namespace gridtools
