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
        GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.i_high_bound() - grid.i_low_bound() + 1;
    }
    template <class GridType, class Grid>
    uint_t block_j_size(backend_ids<target::x86, GridType, strategy::naive> const &, Grid const &grid) {
        GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        return grid.j_high_bound() - grid.j_low_bound() + 1;
    }
} // namespace gridtools
