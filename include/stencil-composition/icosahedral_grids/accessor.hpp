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
        using type = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;
        using location_type = LocationType;
        static const uint_t value = ID;
        using index_type = static_uint< ID >;
        using extent_t = Extent;
        location_type location() const { return location_type(); }

        typedef accessor_base< ID, Intend, Extent, FieldDimensions > super;

/**inheriting all constructors from offset_tuple*/
#ifndef __CUDACC__
        using super::accessor_base;
#else
        // move ctor
        GT_FUNCTION
        constexpr accessor(type &&other) : super(std::move(other)) {}

        // copy ctor
        GT_FUNCTION
        constexpr accessor(type const &other) : super(other) {}
#endif

        GT_FUNCTION
        constexpr accessor() : super() {}

        // copy ctor from an accessor with different ID
        template < uint_t OtherID >
        GT_FUNCTION constexpr accessor(const accessor< OtherID, Intend, LocationType, Extent, FieldDimensions > &other)
            : super(static_cast< const accessor_base< OtherID, Intend, Extent, FieldDimensions > >(other)) {}

        GT_FUNCTION
        constexpr explicit accessor(array< int_t, FieldDimensions > const &offsets) : super(offsets) {}

        template < uint_t Idx >
        GT_FUNCTION constexpr accessor(dimension< Idx > const &x)
            : super(x) {}
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
