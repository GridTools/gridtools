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
#include <boost/mpl/equal.hpp>
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "vector_accessor.hpp"
#include "grid.hpp"
#include "color.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "../extent.hpp"
#include "../esf_aux.hpp"

namespace gridtools {

    template < typename Placeholders, typename Accessors >
    struct are_location_types_compatible {

        template < typename Plc, typename Acc >
        struct same_location_type {
            using type = typename boost::is_same< typename Plc::location_t, typename Acc::location_type >::type;
            static constexpr bool value = type::value;
        };

        static constexpr bool value = boost::mpl::equal< Placeholders,
            Accessors,
            same_location_type< boost::mpl::_1, boost::mpl::_2 > >::type::value;
    };

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ArgSequence, is_arg >::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_stage syntax");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< Grid >::value), "Error: wrong grid type");
        GRIDTOOLS_STATIC_ASSERT((is_color_type< Color >::value), "Error: wrong color type");

        template < uint_t C >
        using esf_function = Functor< C >;
        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = ArgSequence;
        using color_t = Color;

        GRIDTOOLS_STATIC_ASSERT((are_location_types_compatible< args_t, typename Functor< 0 >::arg_list >::value),
            "Location types of placeholders and accessors must match");

        BOOST_MPL_HAS_XXX_TRAIT_DEF(arg_list)
        GRIDTOOLS_STATIC_ASSERT(has_arg_list< esf_function< 0 > >::type::value,
            "The type arg_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'arg_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1, enumtype::in>; \n using v3=accessor<2>; \n using "
            "arg_list=boost::mpl::vector<v1, v3>;");

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef typename impl::make_arg_with_extent_map< args_t, typename esf_function< 0 >::arg_list >::type
            args_with_extents;
    };

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct is_esf_descriptor< esf_descriptor< Functor, Grid, LocationType, Color, ArgSequence > > : boost::mpl::true_ {
    };

    template < typename T >
    struct esf_get_location_type;

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct esf_get_location_type< esf_descriptor< Functor, Grid, LocationType, Color, ArgSequence > > {
        typedef LocationType type;
    };

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence >
    struct esf_descriptor_with_extent : public esf_descriptor< Functor, Grid, LocationType, Color, ArgSequence > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent >::value), "stage descriptor is expecting a extent type");
    };

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence >
    struct is_esf_descriptor< esf_descriptor_with_extent< Functor, Grid, LocationType, Extent, Color, ArgSequence > >
        : boost::mpl::true_ {};

    template < typename ESF >
    struct is_esf_with_extent : boost::mpl::false_ {};

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Extent,
        typename Color,
        typename ArgSequence >
    struct is_esf_with_extent< esf_descriptor_with_extent< Functor, Grid, LocationType, Extent, Color, ArgSequence > >
        : boost::mpl::true_ {};

} // namespace gridtools
