#pragma once
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/arg.hpp"
#include "stencil-composition/esf_fwd.hpp"
#include "stencil-composition/icosahedral_grids/grid.hpp"
#include "vector_accessor.hpp"

#include "../esf_aux.hpp"

namespace gridtools {

    template < typename Functor, typename Grid, typename LocationType, typename ArgSequence >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ArgSequence, is_arg >::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_esf syntax");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< Grid >::value), "Error: wrong grid type");

        using esf_function = Functor;
        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = ArgSequence;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef
            typename impl::make_arg_with_extent_map< args_t, typename esf_function::arg_list >::type args_with_extents;
    };

    template < typename Functor, typename Grid, typename LocationType, typename ArgSequence >
    struct is_esf_descriptor< esf_descriptor< Functor, Grid, LocationType, ArgSequence > > : boost::mpl::true_ {};

    template < typename T >
    struct esf_get_location_type;

    template < typename Functor, typename Grid, typename LocationType, typename ArgSequence >
    struct esf_get_location_type< esf_descriptor< Functor, Grid, LocationType, ArgSequence > > {
        typedef LocationType type;
    };

} // namespace gridtools
