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

#include <array>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"

#include "../backend_ids.hpp"
#include "../coordinate.hpp"
#include "./block.hpp"

namespace gridtools {

    template <class /*StorageInfo*/, class /*MaxExtent*/, enumtype::grid_type GridType, class Grid>
    std::array<uint_t, 3> get_tmp_data_storage_size(
        backend_ids<enumtype::Host, GridType, enumtype::Naive> const &, Grid const &grid) {
        uint_t i_size = grid.direction_i().total_length();
        uint_t j_size = grid.direction_j().total_length();
        uint_t k_size = grid.k_total_length();
        return {i_size, j_size, k_size};
    }

    template <class StorageInfo, class /*MaxExtent*/, enumtype::grid_type GridType, class Grid>
    std::array<uint_t, 3> get_tmp_data_storage_size(
        backend_ids<enumtype::Host, GridType, enumtype::Block> const &backend, Grid const &grid) {
        using halo_t = typename StorageInfo::halo_t;
        using backend_t = backend_ids<enumtype::Host, GridType, enumtype::Block>;
        auto full_block_i_size = block_i_size(backend) + halo_t::template at<coord_i<backend_t>::value>();
        uint_t i_size = full_block_i_size * omp_get_max_threads();
        uint_t j_size = block_j_size(backend) + 2 * halo_t::template at<coord_j<backend_t>::value>();
        uint_t k_size = grid.k_total_length();
        return {i_size, j_size, k_size};
    }

    template <uint_t Coordinate,
        class /*MaxExtent*/,
        class StorageInfo,
        enumtype::grid_type GridType,
        int_t Res = 2 * StorageInfo::halo_t::template at<Coordinate>(),
        size_t I = coord_i<backend_ids<enumtype::Host, GridType, enumtype::Block>>::value>
    constexpr enable_if_t<Coordinate == I, int_t> tmp_storage_block_offset_multiplier(
        backend_ids<enumtype::Host, GridType, enumtype::Block> const &) {
        return Res;
    }

    template <uint_t Coordinate,
        class /*MaxExtent*/,
        class /*StorageInfo*/,
        enumtype::grid_type GridType,
        size_t J = coord_j<backend_ids<enumtype::Host, GridType, enumtype::Block>>::value>
    constexpr enable_if_t<Coordinate == J, int_t> tmp_storage_block_offset_multiplier(
        backend_ids<enumtype::Host, GridType, enumtype::Block> const &backend) {
        return -static_cast<int_t>(block_j_size(backend));
    }
} // namespace gridtools
