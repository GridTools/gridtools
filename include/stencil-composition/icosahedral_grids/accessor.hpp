/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "extent.hpp"

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
