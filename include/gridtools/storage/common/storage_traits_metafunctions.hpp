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

#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/layout_map.hpp"
#include "../../common/selector.hpp"
#include "../../meta/logical.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace _impl {
        /* Layout map extender, takes a given layout and extends it by n dimensions (ascending and descending version)
         */
        template <uint_t Dim, uint_t Current, typename Layout>
        struct layout_map_ext_asc;

        template <uint_t Dim, uint_t Current, int... Dims>
        struct layout_map_ext_asc<Dim, Current, layout_map<Dims...>>
            : layout_map_ext_asc<Dim - 1, Current + 1, layout_map<Dims..., Current>> {};

        template <uint_t Current, int... Dims>
        struct layout_map_ext_asc<0, Current, layout_map<Dims...>> {
            typedef layout_map<Dims...> type;
        };

        template <uint_t Ext, typename Layout>
        struct layout_map_ext_dsc;

        template <uint_t Ext, int... Dims>
        struct layout_map_ext_dsc<Ext, layout_map<Dims...>>
            : layout_map_ext_dsc<Ext - 1, layout_map<Dims..., Ext - 1>> {};

        template <int... Dims>
        struct layout_map_ext_dsc<0, layout_map<Dims...>> {
            typedef layout_map<Dims...> type;
        };
    } // namespace _impl

    /* get a standard layout_map (n-dimensional and ascending or descending) */
    template <uint_t Dim, bool Asc>
    struct get_layout;

    // get a multidimensional layout in ascending order (e.g., host backend)
    /**
     * @brief metafunction used to retrieve a layout_map with n-dimensions
     * that can be used in combination with the host backend (k-first order).
     * E.g., get_layout< 5, true > will return following type: layout_map< 2, 3, 4, 0, 1 >.
     * This means the k-dimension (value: 4) is coalesced in memory, followed
     * by the j-dimension (value: 3), followed by the i-dimension (value: 2), followed
     * by the fifth dimension (value: 1), etc. The reason for having k as innermost
     * is because of the gridtools execution model. The CPU backend will give best
     * performance (in most cases) when using the provided layout.
     */
    template <uint_t Dim>
    struct get_layout<Dim, true> {
        static_assert(Dim > 0, GT_INTERNAL_ERROR_MSG("Zero dimensional layout makes no sense."));
        typedef typename _impl::layout_map_ext_asc<Dim - 3, 0, layout_map<Dim - 3, Dim - 2, Dim - 1>>::type type;
    };

    // get a multidimensional layout in descending order (e.g., gpu backend)
    /**
     * @brief metafunction used to retrieve a layout_map with n-dimensions
     * that can be used in combination with the GPU backend (i-first order).
     * E.g., get_layout< 5, false > will return following type: layout_map< 4, 3, 2, 1, 0 >.
     * This means the i-dimension (value: 4) is coalesced in memory, followed
     * by the j-dimension (value: 3), followed by the k-dimension (value: 2), followed
     * by the fourth dimension (value: 1), etc. The reason for having i as innermost
     * is because of the gridtools execution model. The GPU backend will give best
     * performance (in most cases) when using the provided layout.
     */
    template <uint_t Dim>
    struct get_layout<Dim, false> {
        static_assert(Dim > 0, GT_INTERNAL_ERROR_MSG("Zero dimensional layout makes no sense."));
        typedef typename _impl::layout_map_ext_dsc<Dim - 1, layout_map<Dim - 1>>::type type;
    };

    /* specializations up to 3-dimensional for both i-first and k-first layouts */
    template <>
    struct get_layout<1, true> {
        typedef layout_map<0> type;
    };

    template <>
    struct get_layout<1, false> {
        typedef layout_map<0> type;
    };

    template <>
    struct get_layout<2, true> {
        typedef layout_map<0, 1> type;
    };

    template <>
    struct get_layout<2, false> {
        typedef layout_map<1, 0> type;
    };

    template <>
    struct get_layout<3, true> {
        typedef layout_map<0, 1, 2> type;
    };

    template <>
    struct get_layout<3, false> {
        typedef layout_map<2, 1, 0> type;
    };

    /**
     * @brief metafunction used to retrieve special layout_map.
     * Special layout_map are layout_maps with masked dimensions.
     * @tparam T the layout_map type
     * @tparam Selector the selector type
     */
    template <typename T, typename Selector>
    struct get_special_layout;

    template <int... Dims, bool... Bitmask>
    struct get_special_layout<layout_map<Dims...>, selector<Bitmask...>> {
        static constexpr int correction(int D) {
            return accumulate(plus_functor{}, (!Bitmask && Dims >= 0 && Dims < D ? 1 : 0)...);
        }

        using type = layout_map<(Bitmask ? Dims - correction(Dims) : -1)...>;
    };

    /**
     * @}
     */
} // namespace gridtools
