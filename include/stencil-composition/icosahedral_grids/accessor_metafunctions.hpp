#pragma once

#include "accessor.hpp"

namespace gridtools {

    template < typename Accessor >
    struct accessor_index {
        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Internal Error: wrong type");
        typedef typename Accessor::index_type type;
    };

    template < typename Accessor >
    struct is_accessor_readonly : boost::mpl::false_ {};

    template < int ID, typename LocationType, typename Extent >
    struct is_accessor_readonly< accessor< ID, enumtype::in, LocationType, Extent > > : boost::mpl::true_ {};

    /**
     * @brief metafunction that given an accesor and a map, it will remap the index of the accessor according
     * to the corresponding entry in ArgsMap
     */
    template < typename Accessor, typename ArgsMap >
    struct remap_accessor_type {};

    template < uint_t ID, enumtype::intend Intend, typename LocationType, typename Extent, typename ArgsMap >
    struct remap_accessor_type< accessor< ID, Intend, LocationType, Extent >, ArgsMap > {
        typedef accessor< ID, Intend, LocationType, Extent > accessor_t;
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), "Internal Error: wrong size");
        // check that the key type is an int (otherwise the later has_key would never find the key)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same<
                typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                int >::value),
            "Internal Error");

        typedef typename boost::mpl::integral_c< int, (int)ID > index_type_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_type_t >::value), "Internal Error");

        typedef accessor< boost::mpl::at< ArgsMap, index_type_t >::type::value, Intend, LocationType, Extent > type;
    };

} // namespace gridtools
