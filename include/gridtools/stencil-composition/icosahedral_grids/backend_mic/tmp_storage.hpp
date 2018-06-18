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

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/type_traits.hpp"

#include "../../backend_ids.hpp"
#include "../../backend_traits_fwd.hpp"
#include "../../grid_traits_fwd.hpp"

#include "./strategy_mic.hpp"

namespace gridtools {

    template < class StorageInfo, class /*MaxExtent*/, class Grid >
    std::array< uint_t, 3 > get_tmp_data_storage_size(
        backend_ids< enumtype::Mic, enumtype::icosahedral, enumtype::Block > const &, Grid const &grid) {
        using block_size_t = typename strategy_from_id_mic< enumtype::Block >::block_size_t;
        using grid_traits_t = grid_traits_from_id< enumtype::icosahedral >;
        using halo_t = typename StorageInfo::halo_t;

        static constexpr auto halo_i = halo_t::template at< grid_traits_t::dim_i_t::value >();
        static constexpr auto halo_j = halo_t::template at< grid_traits_t::dim_j_t::value >();
        auto threads = omp_get_max_threads();
        auto i_size = (block_size_t::i_size_t::value + 2 * halo_i) * threads;
        auto j_size = block_size_t::j_size_t::value + 2 * halo_j;
        auto k_size = grid.k_total_length();
        return {i_size, j_size, k_size};
    }

    template < uint_t Coordinate,
        class /*MaxExtent*/,
        class StorageInfo,
        int_t Res = 2 * StorageInfo::halo_t::template at< Coordinate >() >
    constexpr enable_if_t< Coordinate == grid_traits_from_id< enumtype::icosahedral >::dim_i_t::value, int_t >
    tmp_storage_block_offset_multiplier(backend_ids< enumtype::Mic, enumtype::icosahedral, enumtype::Block > const &) {
        return Res;
    }

    template < uint_t Coordinate,
        class /*MaxExtent*/,
        class /*StorageInfo*/,
        int_t Res = -strategy_from_id_mic< enumtype::Block >::block_size_t::j_size_t::value >
    constexpr enable_if_t< Coordinate == grid_traits_from_id< enumtype::icosahedral >::dim_j_t::value, int_t >
    tmp_storage_block_offset_multiplier(backend_ids< enumtype::Mic, enumtype::icosahedral, enumtype::Block > const &) {
        return Res;
    }
}
