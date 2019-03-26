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

/**
 * @file
 *
 * Overloads of block_i_size/block_j_size/block_k_size are defined here.
 *
 * There are two forms of block_*_size :
 *   - GT_FUNCTION constexpr uint_t block_i_size(Backend)
 *   - uint_t block_i_size(Backend, Grid)
 *  where  Backend is an instantiation of backend_ids and Grid is instantiation of grid.
 *
 *  Constexpr form is designed to be used as a part of template parameter or in performance critical code.
 *  The later overload is for backends where the block size depends on runtime. Because it has runtime overhead,
 *  stencil computation framework should call it once at preparation stage and cache the result somewhere.
 *
 *  block_k_size is not used at a moment. That is why it has fallback implementation.
 *
 *  Ideally for backends where block size is compile time, it is enough to define only constexpr version.
 *  And for backends where block size is run time, it is enough to define only the version with two args.
 *  However X86/Naive backend still have to define constexpr version that returns 0.
 *  TODO(anstaf): fix that
 *
 */

#include "../common/defs.hpp"
#include "../common/host_device.hpp"

#include "./grid.hpp"

#include "./backend_cuda/block.hpp"
#include "./backend_naive/block.hpp"
#include "./backend_x86/block.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "./structured_grids/block.hpp"
#endif

namespace gridtools {
    template <class Backend>
    GT_FUNCTION constexpr uint_t block_k_size(Backend const &) {
        return 0;
    }

    template <class Backend, class Grid>
    uint_t block_i_size(Backend const &backend, Grid const &) {
        return block_i_size(backend);
    }
    template <class Backend, class Grid>
    uint_t block_j_size(Backend const &backend, Grid const &) {
        return block_j_size(backend);
    }
    template <class Backend, class Grid>
    uint_t block_k_size(Backend const &, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.k_total_length();
    }
} // namespace gridtools
