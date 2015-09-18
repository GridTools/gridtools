#pragma once
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "stencil-composition/arg.hpp"
#include "stencil-composition/esf_fwd.hpp"
#include "stencil-composition/other_grids/grid.hpp"

namespace gridtools {

template <typename Functor,
          typename Grid,
          typename LocationType,
          typename ArgSequence>
struct esf_descriptor
{
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ArgSequence, is_arg>::value), "wrong types for the list of parameter placeholders\n"
            "check the make_esf syntax");
    GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Error: wrong grid type");

    using esf_function = Functor;
    using grid = Grid;
    using location_type = LocationType;
    using args_t = ArgSequence;
};

template<typename Functor, typename Grid, typename LocationType, typename ArgSequence>
struct is_esf_descriptor<esf_descriptor<Functor, Grid, LocationType, ArgSequence> > : boost::mpl::true_{};

} // namespace gridtools
