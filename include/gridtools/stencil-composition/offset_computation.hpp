/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/mpl/eval_if.hpp>

#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/gt_assert.hpp"
#include "../meta/type_traits.hpp"
#include "../meta/utility.hpp"

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
    template <typename StorageInfo, int_t Coordinate, typename StridesCached>
    GT_FUNCTION constexpr int_t stride(StridesCached const &GT_RESTRICT strides) {
        using layout_t = typename StorageInfo::layout_t;

        /* get the maximum integer value in the layout map */
        using layout_max_t = std::integral_constant<int, layout_t::max()>;

        /* get the layout map value at the given coordinate */
        using layout_val_t = std::integral_constant<int, layout_t::template at_unsafe<Coordinate>()>;

        /* check if we are at a masked-out value (-> stride == 0) */
        using is_masked_t = bool_constant<layout_val_t::value == -1>;
        /* check if we are at the maximum value (-> stride == 1) */
        using is_max_t = bool_constant<layout_max_t::value == layout_val_t::value>;

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
    template <int_t Coordinate, typename Accessor>
    GT_FUNCTION constexpr int_t accessor_offset(Accessor const &accessor) {
        return get<Coordinate>(accessor);
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
        template <typename StorageInfo, typename StridesCached, typename Accessor, std::size_t... Coordinates>
        GT_FUNCTION constexpr int_t compute_offset(StridesCached const &GT_RESTRICT strides,
            Accessor const &GT_RESTRICT accessor,
            meta::index_sequence<Coordinates...>) {
            /* sum stride_x * offset_x + stride_y * offset_y + ... */
            return accumulate(plus_functor(),
                (stride<StorageInfo, Coordinates>(strides) * accessor_offset<Coordinates>(accessor))...);
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
    template <typename StorageInfo, typename Accessor, typename StridesCached>
    GT_FUNCTION constexpr int_t compute_offset(
        StridesCached const &GT_RESTRICT strides, Accessor const &GT_RESTRICT accessor) {
        using sequence_t = meta::make_index_sequence<StorageInfo::layout_t::masked_length>;
        return _impl::compute_offset<StorageInfo>(strides, accessor, sequence_t());
    }
} // namespace gridtools
