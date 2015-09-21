#pragma once
#include "radius.hpp"

namespace gridtools {
/**
   This is the type of the accessors accessed by a stencil functor.
   It's a pretty minima implementation.
 */
template <int I, typename LocationType>
struct accessor {
    GRIDTOOLS_STATIC_ASSERT((is_location_type<LocationType>::value), "Error: wrong type");
    using this_type = accessor<I, LocationType>;
    using location_type = LocationType;
    static const int value = I;

    location_type location() const {
        return location_type();
    }
};

template <int I, typename LocationType, typename Radius=radius<0> >
struct ro_accessor : public accessor<I, LocationType> {
    using radius_type = Radius;
};

template <typename T>
struct is_accessor : boost::mpl::false_{};

template <int I, typename LocationType>
struct is_accessor<accessor<I, LocationType> > : boost::mpl::true_{};

/**
 * Struct to test if an argument is a temporary
 */
//TODO for the moment we dont support temporaries
template <typename T>
struct is_plchldr_to_temp : boost::mpl::false_{};


} //namespace gridtools
