#pragma once
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/arg.hpp"
#include "stencil-composition/esf_fwd.hpp"
#include "stencil-composition/icosahedral_grids/grid.hpp"
#include "../esf_aux.hpp"
#include "color.hpp"

namespace gridtools {

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ArgSequence, is_arg >::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_esf syntax");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< Grid >::value), "Error: wrong grid type");
        GRIDTOOLS_STATIC_ASSERT((is_color_type< Color >::value), "Error: wrong color type");

        template < uint_t C >
        using esf_function = Functor< C >;
        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = ArgSequence;
        using color_t = Color;

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

} // namespace gridtools
