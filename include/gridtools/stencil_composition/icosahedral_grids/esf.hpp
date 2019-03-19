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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "../extent.hpp"
#include "color.hpp"
#include "icosahedral_topology.hpp"

namespace gridtools {
    namespace esf_impl_ {
        template <class Arg, class Accessor>
        GT_META_DEFINE_ALIAS(
            is_same_location, std::is_same, (typename Arg::location_t, typename Accessor::location_type));

        template <class Args, class Accessors>
        GT_META_DEFINE_ALIAS(
            are_same_locations, meta::all, (GT_META_CALL(meta::transform, (is_same_location, Args, Accessors))));

        template <class, class = void>
        struct has_param_list : std::false_type {};

        template <class T>
        struct has_param_list<T, void_t<typename T::param_list>> : std::true_type {};
    } // namespace esf_impl_

    template <template <uint_t> class EsfFunction, class Grid, class LocationType, class Color, class Args>
    struct esf_descriptor {
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value),
            "wrong types for the list of parameter placeholders check the make_stage syntax");
        GT_STATIC_ASSERT((is_sequence_of<Args, is_plh>::value),
            "wrong types for the list of parameter placeholders check the make_stage syntax");
        GT_STATIC_ASSERT(is_grid_topology<Grid>::value, "Error: wrong grid type");
        GT_STATIC_ASSERT(is_color_type<Color>::value, "Error: wrong color type");
        GT_STATIC_ASSERT((esf_impl_::are_same_locations<Args, typename EsfFunction<0>::param_list>::value),
            "Location types of placeholders and accessors must match");
        GT_STATIC_ASSERT(esf_impl_::has_param_list<EsfFunction<0>>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\'.");

        template <uint_t C>
        using esf_function = EsfFunction<C>;

        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = Args;
        using color_t = Color;
    };

    template <template <uint_t> class EsfFunction, class Grid, class LocationType, class Color, class Args>
    struct is_esf_descriptor<esf_descriptor<EsfFunction, Grid, LocationType, Color, Args>> : std::true_type {};

    template <class T>
    struct esf_get_location_type;

    template <template <uint_t> class EsfFunction, class Grid, class LocationType, class Color, class Args>
    struct esf_get_location_type<esf_descriptor<EsfFunction, Grid, LocationType, Color, Args>> {
        using type = LocationType;
    };
} // namespace gridtools
