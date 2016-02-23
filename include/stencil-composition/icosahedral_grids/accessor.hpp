#pragma once
#include "radius.hpp"

namespace gridtools {
    /**
    * This is the type of the accessors accessed by a stencil functor.
    * It's a pretty minima implementation.
    */
    template <uint_t ID, enumtype::intend Intend, typename LocationType, typename Radius=radius<0> >
    struct accessor {
        GRIDTOOLS_STATIC_ASSERT((is_location_type<LocationType>::value), "Error: wrong type");
        using type = accessor<ID, Intend, LocationType, Radius>;
        using location_type = LocationType;
        static const uint_t value = ID;
        using index_type = static_uint<ID>;

        location_type location() const {
            return location_type();
        }
    };

    template<uint_t ID, typename LocationType, typename Radius=radius<0> >
    using in_accessor = accessor<ID, enumtype::in, LocationType, Radius>;

    template<uint_t ID, typename LocationType>
    using inout_accessor = accessor<ID, enumtype::inout, LocationType, radius<0>>;

    template<typename T>
    struct is_accessor : boost::mpl::false_{};

    template <uint_t ID, enumtype::intend Intend, typename LocationType, typename Radius>
    struct is_accessor<accessor<ID, Intend, LocationType, Radius> > : boost::mpl::true_{};

    /**
    * Struct to test if an argument is a temporary
    */
    //TODO for the moment we dont support temporaries
    template <typename T>
    struct is_plchldr_to_temp : boost::mpl::false_{};
} //namespace gridtools
