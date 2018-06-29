/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
 *  However Host/Naive backend still have to define constexpr version that returns 0.
 *  TODO(anstaf): fix that
 *
 */

#include "../common/defs.hpp"
#include "../common/host_device.hpp"

#include "./grid.hpp"

#include "./backend_cuda/block.hpp"
#include "./backend_host/block.hpp"

#ifdef STRUCTURED_GRIDS
#include "./structured_grids/block.hpp"
#else
#include "./icosahedral_grids/block.hpp"
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
        GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.k_total_length();
    }
} // namespace gridtools
