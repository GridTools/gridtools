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

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../backend_ids.hpp"
#include "./execinfo_mc.hpp"

namespace gridtools {
    template <class Grid>
    uint_t block_i_size(backend_ids<target::mc, grid_type::structured, strategy::block> const &, Grid const &grid) {
        return execinfo_mc{grid}.i_block_size();
    }
    template <class Grid>
    uint_t block_j_size(backend_ids<target::mc, grid_type::structured, strategy::block> const &, Grid const &grid) {
        return execinfo_mc{grid}.j_block_size();
    }
} // namespace gridtools
