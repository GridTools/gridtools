#pragma once

#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_integral.hpp>
#include "array.hpp"

namespace gridtools {

    /**
     * type trait to check if a type is an aggregate
     * Note: see discussion here
     * http://stackoverflow.com/questions/33648044/boostprotois-aggregate-returning-false-when-it-is-an-aggregate-type
     * there is not general way of detecting whether a type in C++ is an aggregate, and there probably wont be.
     * Instead we use specific traits for the types that are used in our library
     * (in the future this might be extended to using concepts)
     */
    template < typename T >
    struct is_aggregate : boost::mpl::or_< is_array< T >, boost::is_integral< T > > {};
}
