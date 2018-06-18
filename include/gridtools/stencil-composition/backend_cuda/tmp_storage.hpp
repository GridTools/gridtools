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
#include "../backend_traits_fwd.hpp"
#include "../grid_traits_fwd.hpp"

#include "./strategy_cuda.hpp"

namespace gridtools {

    namespace _impl {
        template < class StorageInfo,
            size_t alignment = StorageInfo::alignment_t::value ? StorageInfo::alignment_t::value : 1 >
        static constexpr size_t align(size_t x) {
            return (x + alignment - 1) / alignment * alignment;
        }
    }

    template < class StorageInfo, class MaxExtent, enumtype::grid_type GridType, class Grid >
    std::array< uint_t, 3 > get_tmp_data_storage_size(
        backend_ids< enumtype::Cuda, GridType, enumtype::Block > const &, Grid const &grid) {
        GRIDTOOLS_STATIC_ASSERT(is_grid< Grid >::value, GT_INTERNAL_ERROR);
        using block_size_t = typename strategy_from_id_cuda< enumtype::Block >::block_size_t;
        using grid_traits_t = grid_traits_from_id< GridType >;

        static constexpr uint_t full_block_i_size =
            _impl::align< StorageInfo >(block_size_t::i_size_t::value + 2 * MaxExtent::value);
        static constexpr uint_t full_block_j_size =
            block_size_t::j_size_t::value + 2 * StorageInfo::halo_t::template at< grid_traits_t::dim_j_t::value >();

        auto k_size = grid.k_total_length();
        auto num_blocks_i =
            (grid.i_high_bound() - grid.i_low_bound() + block_size_t::i_size_t::value) / block_size_t::i_size_t::value;
        auto num_blocks_j =
            (grid.j_high_bound() - grid.j_low_bound() + block_size_t::j_size_t::value) / block_size_t::j_size_t::value;
        return {full_block_i_size * num_blocks_i, full_block_j_size * num_blocks_j, k_size};
    }

    template < uint_t Coordinate,
        class MaxExtent,
        class StorageInfo,
        enumtype::grid_type GridType,
        size_t BlockSize = strategy_from_id_cuda< enumtype::Block >::block_size_t::i_size_t::value,
        size_t FullBlockSize = _impl::align< StorageInfo >(BlockSize + 2 * MaxExtent::value),
        int_t Res = FullBlockSize - BlockSize >
    GT_FUNCTION constexpr enable_if_t< Coordinate == grid_traits_from_id< GridType >::dim_i_t::value, int_t >
    tmp_storage_block_offset_multiplier(backend_ids< enumtype::Cuda, GridType, enumtype::Block > const &) {
        return Res;
    }

    template < uint_t Coordinate,
        class MaxExtent,
        class StorageInfo,
        enumtype::grid_type GridType,
        int_t Res = 2 * StorageInfo::halo_t::template at< Coordinate >() >
    GT_FUNCTION constexpr enable_if_t< Coordinate == grid_traits_from_id< GridType >::dim_j_t::value, int_t >
    tmp_storage_block_offset_multiplier(backend_ids< enumtype::Cuda, GridType, enumtype::Block > const &) {
        return Res;
    }
}
