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
#include "../../common/host_device.hpp"

#include "../backend_ids.hpp"
#include "../coordinate.hpp"
#include "../grid.hpp"
#include "./block.hpp"

namespace gridtools {

    namespace _impl {
        template <class StorageInfo,
            size_t alignment = StorageInfo::alignment_t::value ? StorageInfo::alignment_t::value : 1>
        GT_FUNCTION constexpr size_t align(size_t x) {
            return (x + alignment - 1) / alignment * alignment;
        }
    } // namespace _impl

    template <class StorageInfo, class MaxExtent, enumtype::grid_type GridType, class Grid>
    std::array<uint_t, 3> get_tmp_data_storage_size(
        backend_ids<enumtype::Cuda, GridType, enumtype::Block> const &backend, Grid const &grid) {
        GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using backend_t = backend_ids<enumtype::Cuda, GridType, enumtype::Block>;

        auto pe_block_i_size = block_i_size(backend);
        auto pe_block_j_size = block_j_size(backend);

        uint_t full_block_i_size = _impl::align<StorageInfo>(pe_block_i_size + 2 * MaxExtent::value);
        uint_t full_block_j_size = pe_block_j_size + 2 * StorageInfo::halo_t::template at<coord_j<backend_t>::value>();

        uint_t k_size = grid.k_total_length();
        uint_t num_blocks_i = (grid.i_high_bound() - grid.i_low_bound() + pe_block_i_size) / pe_block_i_size;
        uint_t num_blocks_j = (grid.j_high_bound() - grid.j_low_bound() + pe_block_j_size) / pe_block_j_size;
        return {full_block_i_size * num_blocks_i, full_block_j_size * num_blocks_j, k_size};
    }

    template <uint_t Coordinate,
        class MaxExtent,
        class StorageInfo,
        enumtype::grid_type GridType,
        enable_if_t<Coordinate == coord_i<backend_ids<enumtype::Cuda, GridType, enumtype::Block>>::value, int> = 0>
    GT_FUNCTION constexpr int_t tmp_storage_block_offset_multiplier(
        backend_ids<enumtype::Cuda, GridType, enumtype::Block> const &backend) {
        return _impl::align<StorageInfo>(block_i_size(backend) + 2 * MaxExtent::value) - block_i_size(backend);
    }

    template <uint_t Coordinate,
        class MaxExtent,
        class StorageInfo,
        enumtype::grid_type GridType,
        enable_if_t<Coordinate == coord_j<backend_ids<enumtype::Cuda, GridType, enumtype::Block>>::value, int> = 0>
    GT_FUNCTION constexpr int_t tmp_storage_block_offset_multiplier(
        backend_ids<enumtype::Cuda, GridType, enumtype::Block> const &) {
        return 2 * StorageInfo::halo_t::template at<Coordinate>();
    }
} // namespace gridtools
