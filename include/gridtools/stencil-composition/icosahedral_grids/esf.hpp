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
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../arg.hpp"
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "../extent.hpp"
#include "color.hpp"
#include "grid.hpp"
#include "icosahedral_topology.hpp"
#include <boost/mpl/equal.hpp>

namespace gridtools {

    template <typename Placeholders, typename Accessors>
    struct are_location_types_compatible {

        template <typename Plc, typename Acc>
        struct same_location_type {
            using type = typename boost::is_same<typename Plc::location_t, typename Acc::location_type>::type;
            static constexpr bool value = type::value;
        };

        static constexpr bool value =
            boost::mpl::equal<Placeholders, Accessors, same_location_type<boost::mpl::_1, boost::mpl::_2>>::type::value;
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence>
    struct esf_descriptor {
        GT_STATIC_ASSERT((is_sequence_of<ArgSequence, is_plh>::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_stage syntax");
        GT_STATIC_ASSERT((is_grid_topology<Grid>::value), "Error: wrong grid type");
        GT_STATIC_ASSERT((is_color_type<Color>::value), "Error: wrong color type");

        template <uint_t C>
        using esf_function = Functor<C>;
        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = ArgSequence;
        using color_t = Color;

        GT_STATIC_ASSERT((are_location_types_compatible<args_t, typename Functor<0>::param_list>::value),
            "Location types of placeholders and accessors must match");

        BOOST_MPL_HAS_XXX_TRAIT_DEF(param_list)
        GT_STATIC_ASSERT(has_param_list<esf_function<0>>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1>; \n using v3=accessor<2>; \n using "
            "param_list=boost::mpl::vector<v1, v3>;");

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        using args_with_extents_t =
            typename impl::make_arg_with_extent_map<args_t, typename esf_function<0>::param_list>::type;
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence>
    struct is_esf_descriptor<esf_descriptor<Functor, Grid, LocationType, Color, ArgSequence>> : boost::mpl::true_ {};

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
    struct esf_descriptor_with_extent : public esf_descriptor<Functor, Grid, LocationType, Color, ArgSequence> {
        GT_STATIC_ASSERT((is_extent<Extent>::value), "stage descriptor is expecting a extent type");
    };

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct is_esf_descriptor<esf_descriptor_with_extent<Functor, Grid, LocationType, Extent, Color, ArgSequence>>
        : boost::mpl::true_ {};

    template <typename ESF>
    struct is_esf_with_extent : boost::mpl::false_ {};

    template <template <uint_t> class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence>
    struct is_esf_with_extent<esf_descriptor_with_extent<Functor, Grid, LocationType, Extent, Color, ArgSequence>>
        : boost::mpl::true_ {};

} // namespace gridtools
