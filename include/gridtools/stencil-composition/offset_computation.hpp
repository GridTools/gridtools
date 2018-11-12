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

#include <boost/mpl/eval_if.hpp>

#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/gt_assert.hpp"

namespace gridtools {
    /**
     * This function gets the accessor offset along the the given axis/coordinate.
     *
     * @tparam Coordinate The axis for which the offset should be returned.
     * @tparam Accessor Type of the accessor.
     *
     * @param accessor Accessor for which the offsets should be returned.
     *
     * @return The offset stored in the given accessor for the given axis.
     */
    template <int_t Coordinate, typename Accessor>
    GT_FUNCTION constexpr int_t accessor_offset(Accessor const &accessor) {
        return get<Coordinate>(accessor);
    }

    template <int_t Coord, class T>
    int_t get_stride(T);

    namespace _impl {
        /**
         * This function computes the total accessor-induces pointer offset for multiple axes for a storage.
         *
         * @tparam A type on which get<Coordinate> returns the stride in direction Coordinate.
         * @tparam Accessor Type of the accessor.
         * @tparam Coordinates The axes along which the offsets should be accumulated.
         *
         * @param accessor Accessor for which the offsets should be computed.
         *
         * @return The data offset computed for the given storage info and accessor for the given axes.
         */
        template <typename StridesGetter, typename Accessor, std::size_t... Coordinates>
        GT_FUNCTION constexpr int_t compute_offset(StridesGetter const &RESTRICT strides_getter,
            Accessor const &RESTRICT accessor,
            gt_index_sequence<Coordinates...>) {
            /* sum stride_x * offset_x + stride_y * offset_y + ... */
            return accumulate(
                plus_functor(), (get_stride<Coordinates>(strides_getter) * accessor_offset<Coordinates>(accessor))...);
        }
    } // namespace _impl

    /**
     * This function computes the total accessor-induces pointer offset (sum) for all axes in the given storage info.
     *
     * @tparam StorageInfo The storage info to be used.
     * @tparam StridesCached Type of the strides array.
     * @tparam Accessor Type of the accessor.
     *
     * @param strides Array of stride values.
     * @param accessor Accessor for which the offsets should be computed.
     *
     * @return The total data offset computed for the given storage info and accessor.
     */
    template <typename StridesWrapper, typename Accessor>
    GT_FUNCTION constexpr int_t compute_offset(
        StridesWrapper const &RESTRICT strides_wrapper, Accessor const &RESTRICT accessor) {
        using sequence_t =
            make_gt_index_sequence<decltype(get_layout_map(std::declval<StridesWrapper>()))::masked_length>;
        return _impl::compute_offset(strides_wrapper, accessor, sequence_t{});
    }
} // namespace gridtools
