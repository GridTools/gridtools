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
#include "position_offset_type.hpp"

namespace gridtools {

    /**
     * This function computes the stride along the the given axis/coordinate.
     *
     * @tparam StorageInfo The storage info to be used.
     * @tparam Coordinate The axis along which the stride should be computed.
     * @tparam StridesCached Type of the strides array.
     *
     * @param strides Array of stride values.
     *
     * @return The stride along the given axis: 0 if the axis is masked, 1 if the axis is the contiguous one, run
     * time value read from `strides` otherwise.
     */
    template < typename StorageInfo, int_t Coordinate, typename StridesCached >
    GT_FUNCTION constexpr int_t stride(StridesCached const &RESTRICT strides) {
        using layout_t = typename StorageInfo::layout_t;

        /* get the maximum integer value in the layout map */
        using layout_max_t = std::integral_constant< int, layout_t::max() >;

        /* get the layout map value at the given coordinate */
        using layout_val_t = std::integral_constant< int, layout_t::template at_unsafe< Coordinate >() >;

        /* check if we are at a masked-out value (-> stride == 0) */
        using is_masked_t = bool_constant< layout_val_t::value == -1 >;
        /* check if we are at the maximum value (-> stride == 1) */
        using is_max_t = bool_constant< layout_max_t::value == layout_val_t::value >;

        /* return constants for masked and max coordinates, otherwise lookup stride */
        return is_masked_t::value ? 0 : is_max_t::value ? 1 : strides[layout_val_t::value];
    }

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
    template < int_t Coordinate, typename Accessor >
    GT_FUNCTION constexpr typename std::enable_if< !is_position_offset_type< Accessor >::value, int_t >::type
    accessor_offset(Accessor const &accessor) {
        return accessor.template get< Accessor::n_dimensions - 1 - Coordinate >();
    }

    template < int_t Coordinate, typename Accessor >
    GT_FUNCTION constexpr typename std::enable_if< is_position_offset_type< Accessor >::value, int_t >::type
    accessor_offset(Accessor const &accessor) {
        return accessor.template get< Coordinate >();
    }

    namespace _impl {

        /**
         * This function computes the accessor-induces pointer offset (sum) for multiple axes.
         *
         * @tparam StorageInfo The storage info to be used.
         * @tparam StridesCached Type of the strides array.
         * @tparam Accessor Type of the accessor.
         * @tparam Coordinates The axes along which the offsets should be accumulated.
         *
         * @param strides Array of stride values.
         * @param accessor Accessor for which the offsets should be computed.
         *
         * @return The data offset computed for the given storage info and accessor for the given axes.
         */
        template < typename StorageInfo, typename StridesCached, typename Accessor, std::size_t... Coordinates >
        GT_FUNCTION constexpr int_t compute_offset(StridesCached const &RESTRICT strides,
            Accessor const &RESTRICT accessor,
            gt_index_sequence< Coordinates... >) {
            /* sum stride_x * offset_x + stride_y * offset_y + ... */
            return accumulate(plus_functor(),
                (stride< StorageInfo, Coordinates >(strides) * accessor_offset< Coordinates >(accessor))...);
        }

        /**
         * This function computes the total accessor-induces pointer offset for multiple axes for a cache storage.
         *
         * @tparam StorageInfo The storage info to be used.
         * @tparam Accessor Type of the accessor.
         * @tparam Coordinates The axes along which the offsets should be accumulated.
         *
         * @param accessor Accessor for which the offsets should be computed.
         *
         * @return The data offset computed for the given storage info and accessor for the given axes.
         */
        template < typename StorageInfo, typename Accessor, std::size_t... Coordinates >
        GT_FUNCTION constexpr int_t compute_offset_cache(
            Accessor const &RESTRICT accessor, gt_index_sequence< Coordinates... >) {
            return accumulate(plus_functor(),
                (StorageInfo::template stride< Coordinates >() * accessor_offset< Coordinates >(accessor))...);
        }
    }

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
    template < typename StorageInfo, typename Accessor, typename StridesCached >
    GT_FUNCTION constexpr int_t compute_offset(
        StridesCached const &RESTRICT strides, Accessor const &RESTRICT accessor) {
        using sequence_t = make_gt_index_sequence< StorageInfo::layout_t::masked_length >;
        return _impl::compute_offset< StorageInfo >(strides, accessor, sequence_t());
    }

    /**
     * This function computes the total accessor-induces pointer offset (sum) for all axes in the given cache storage
     * info.
     *
     * @tparam StorageInfo The storage info to be used.
     * @tparam Accessor Type of the accessor.
     *
     * @param accessor Accessor for which the offsets should be computed.
     *
     * @return The total data offset computed for the given storage info and accessor.
     */
    template < typename StorageInfo, typename Accessor >
    GT_FUNCTION constexpr int_t compute_offset_cache(Accessor const &accessor) {
        using sequence_t = make_gt_index_sequence< StorageInfo::layout_t::masked_length >;
        return _impl::compute_offset_cache< StorageInfo >(accessor, sequence_t());
    }
}
