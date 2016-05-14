#pragma once
#include "extent.hpp"
#include "../accessor_base.hpp"

namespace gridtools {
    /**
    * This is the type of the accessors accessed by a stencil functor.
    * It's a pretty minima implementation.
    */
    template < uint_t ID,
        enumtype::intend Intend,
        typename LocationType,
        typename Extent = extent< 0 >,
        ushort_t FieldDimensions = 4 >
    struct accessor : public accessor_base< ID, Intend, Extent, FieldDimensions > {
        GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "Error: wrong type");
        using type = accessor< ID, Intend, LocationType, Extent >;
        using location_type = LocationType;
        static const uint_t value = ID;
        using index_type = static_uint< ID >;
        using extent_t = Extent;
        location_type location() const { return location_type(); }

        typedef accessor_base< ID, Intend, Extent, FieldDimensions > super;

        /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;
    };

    template < uint_t ID, typename LocationType, typename Extent = extent< 0 >, ushort_t FieldDimensions = 4 >
    using in_accessor = accessor< ID, enumtype::in, LocationType, Extent, FieldDimensions >;

    template < uint_t ID, typename LocationType, ushort_t FieldDimensions = 4 >
    using inout_accessor = accessor< ID, enumtype::inout, LocationType, extent< 0 >, FieldDimensions >;

    template < typename T >
    struct is_accessor : boost::mpl::false_ {};

    template < uint_t ID, enumtype::intend Intend, typename LocationType, typename Extent, ushort_t FieldDimensions >
    struct is_accessor< accessor< ID, Intend, LocationType, Extent, FieldDimensions > > : boost::mpl::true_ {};

} // namespace gridtools
