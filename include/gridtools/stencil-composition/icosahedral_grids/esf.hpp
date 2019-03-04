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
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "../extent.hpp"
#include "color.hpp"
#include "grid.hpp"
#include "icosahedral_topology.hpp"

namespace gridtools {
    namespace esf_impl_ {
        template <typename Arg, typename Accessor>
        GT_META_DEFINE_ALIAS(
            is_same_location, std::is_same, (typename Arg::location_t, typename Accessor::location_type));

        template <typename Args, typename Accessors>
        GT_META_DEFINE_ALIAS(
            are_same_locations, meta::all, (GT_META_CALL(meta::transform, (is_same_location, Args, Accessors))));

        template <class, class = void>
        struct has_param_list : std::false_type {};

        template <class T>
        struct has_param_list<T, void_t<typename T::param_list>> : std::true_type {};
    } // namespace esf_impl_

    template <template <uint_t> class EsfFunction, typename Grid, typename LocationType, typename Color, typename Args>
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
            "called \'param_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1>; \n using v3=accessor<2>; \n using "
            "param_list=make_param_list<v1, v3>;");

        template <uint_t C>
        using esf_function = EsfFunction<C>;

        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = Args;
        using color_t = Color;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        using args_with_extents_t =
            typename impl::make_arg_with_extent_map<args_t, typename EsfFunction<0>::param_list>::type;
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence>
    struct is_esf_descriptor<esf_descriptor<Functor, Grid, LocationType, Color, ArgSequence>> : std::true_type {};

    template <typename T>
    struct esf_get_location_type;

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence>
    struct esf_get_location_type<esf_descriptor<Functor, Grid, LocationType, Color, ArgSequence>> {
        typedef LocationType type;
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct esf_descriptor_with_extent : esf_descriptor<Functor, Grid, LocationType, Color, ArgSequence> {
        GT_STATIC_ASSERT(is_extent<Extent>::value, "stage descriptor is expecting a extent type");
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct is_esf_descriptor<esf_descriptor_with_extent<Functor, Grid, LocationType, Extent, Color, ArgSequence>>
        : std::true_type {};

    template <typename>
    struct is_esf_with_extent : std::false_type {};

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct is_esf_with_extent<esf_descriptor_with_extent<Functor, Grid, LocationType, Extent, Color, ArgSequence>>
        : std::true_type {};

} // namespace gridtools
