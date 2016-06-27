#pragma once
#include "extent.hpp"
#include "location_type.hpp"

namespace gridtools {
    /**
    * This is the type of the accessors accessed by a stencil functor.
    * It's a pretty minima implementation.
    */
    template < uint_t ID, enumtype::intend Intend, typename LocationType, typename Extent = extent< 0 > >
    struct accessor {
        GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "Error: wrong type");
        using type = accessor< ID, Intend, LocationType, Extent >;
        using location_type = LocationType;
        static const uint_t value = ID;
        using index_type = static_uint< ID >;
        using extent_t = Extent;
        location_type location() const { return location_type(); }
    };

    template < uint_t ID, typename LocationType, typename Extent = extent< 0 > >
    using in_accessor = accessor< ID, enumtype::in, LocationType, Extent >;

    template < uint_t ID, typename LocationType >
    using inout_accessor = accessor< ID, enumtype::inout, LocationType, extent< 0 > >;

    template < typename T >
    struct is_accessor : boost::mpl::false_ {};

    template < uint_t ID, enumtype::intend Intend, typename LocationType, typename Extent >
    struct is_accessor< accessor< ID, Intend, LocationType, Extent > > : boost::mpl::true_ {};

    /**
    * Struct to test if an argument is a temporary
    */
    // TODO for the moment we dont support temporaries
    template < typename T >
    struct is_plchldr_to_temp : boost::mpl::false_ {};
} // namespace gridtools
