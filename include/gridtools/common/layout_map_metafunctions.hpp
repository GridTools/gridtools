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

#include <utility>

#include "layout_map.hpp"
#include "selector.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \ingroup layout
        @{
    */

    /** \brief Compute the reverse of a layout. For instance the reverse of
        `layout_map<1,0,-1,2>` is `layout<1,2,-1,0>`

        \tparam LayoutMap The layout map to process
    */
    template <typename LayoutMap>
    struct reverse_map;

    /// \private
    template <int_t... Is>
    struct reverse_map<layout_map<Is...>> {
        static constexpr int max = layout_map<Is...>::max();
        using type = layout_map<(Is < 0 ? Is : max - Is)...>;
    };

    template <typename DATALO, typename PROCLO>
    struct layout_transform;

    template <class Layout, int_t... P>
    struct layout_transform<Layout, layout_map<P...>> {
        using type = layout_map<Layout::template at<P>()...>;
    };

    enum class insert_location { pre, post };

    template <typename LayoutMap, int_t NExtraDim, insert_location Location = insert_location::post>
    struct extend_layout_map;

    /*
     * metafunction to extend a layout_map with certain number of dimensions.
     * Example of use:
     * a) extend_layout_map< layout_map<0, 1, 3, 2>, 3> == layout_map<3, 4, 6, 5, 0, 1, 2>
     * b) extend_layout_map< layout_map<0, 1, 3, 2>, 3, insert_location::pre> == layout_map<0, 1, 2, 3, 4, 6, 5>
     */
    template <int_t NExtraDim, int_t... Args, insert_location Location>
    struct extend_layout_map<layout_map<Args...>, NExtraDim, Location> {

        template <insert_location Loc, typename Seq>
        struct build_ext_layout;

        // build an extended layout
        template <int_t... Indices>
        struct build_ext_layout<insert_location::post, std::integer_sequence<int_t, Indices...>> {
            using type = layout_map<(Args == -1 ? -1 : Args + NExtraDim)..., Indices...>;
        };
        template <int_t... Indices>
        struct build_ext_layout<insert_location::pre, std::integer_sequence<int_t, Indices...>> {
            using type = layout_map<Indices..., (Args == -1 ? -1 : Args + NExtraDim)...>;
        };

        using type = typename build_ext_layout<Location, std::make_integer_sequence<int_t, NExtraDim>>::type;
    };

    template <int_t D>
    struct default_layout_map {
        using type = typename extend_layout_map<layout_map<>, D, insert_location::pre>::type;
    };

    template <int_t D>
    using default_layout_map_t = typename default_layout_map<D>::type;

    /** @} */
    /** @} */
} // namespace gridtools
