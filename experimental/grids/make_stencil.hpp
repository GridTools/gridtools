#pragma once
#include <boost/mpl/vector.hpp>

template <typename Functor,
          typename Grid,
          typename LocationType,
          typename PlcHldrs>
struct esf
{
    using functor = Functor;
    using grid = Grid;
    using location_type = LocationType;
    using plcs = PlcHldrs;
};


template <typename Functor,
          typename Grid,
          typename LocationType,
          typename ...PlcHldr0>
esf<Functor, Grid, LocationType, boost::mpl::vector<PlcHldr0...> >
make_esf(PlcHldr0... args) {
    return esf<Functor, Grid, LocationType, boost::mpl::vector<PlcHldr0...>>();
}

