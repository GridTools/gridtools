#pragma once

namespace gridtools {

    template <typename T>
    struct is_array;

    template <typename T>
    struct is_offset_tuple;

    // metafunction determines whether an argument is an offset_tuple or an array
    template < typename T >
    struct is_tuple_or_array
        : boost::mpl::or_< typename is_offset_tuple< T >::type, typename is_array< T >::type >::type {};

} // namespace gridtools
