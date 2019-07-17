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
 *  where  Backend is instantiation of the backend target tag and Grid is instantiation of grid.
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

#include "backend_x86/block.hpp"

namespace gridtools {
    template <class Backend, class Grid>
    int_t block_i_size(Backend, Grid &&) {
        return block_i_size(Backend());
    }
    template <class Backend, class Grid>
    int_t block_j_size(Backend, Grid &&) {
        return block_j_size(Backend());
    }
} // namespace gridtools
