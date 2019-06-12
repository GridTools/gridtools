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

namespace gridtools {
    GT_FUNCTION constexpr int_t block_i_size(backend::cuda const &) { return GT_DEFAULT_TILE_I; }
    GT_FUNCTION constexpr int_t block_j_size(backend::cuda const &) { return GT_DEFAULT_TILE_J; }
} // namespace gridtools
