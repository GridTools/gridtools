#pragma once
#include <boost/mpl/vector.hpp>
#include <boost/mpl/push_back.hpp>

namespace gridtools {

#ifdef CXX11_ENABLED
    /**
     * @struct variadic_to_vector
     * metafunction that returns a mpl vector from a pack of variadic arguments
     * This is a replacement of using type=boost::mpl::vector<Args ...>, but at the moment nvcc
     * does not properly unpack the last arg of Args... when building the vector. We can eliminate this
     * metafunction once the vector<Args...> works
     */
    template < typename... Args >
    struct variadic_to_vector;

    template < class T, typename... Args >
    struct variadic_to_vector< T, Args... > {
        typedef typename boost::mpl::push_front< typename variadic_to_vector< Args... >::type, T >::type type;
    };

    template < class T >
    struct variadic_to_vector< T > {
        typedef boost::mpl::vector< T > type;
    };

    template <>
    struct variadic_to_vector<> {
        typedef boost::mpl::vector<> type;
    };

#endif
}
