#pragma once
#include <boost/mpl/not.hpp>

namespace gridtools {

    /*
    * @struct is_not_same
    * just a not of is_same
    */
    template < typename T1, typename T2 >
    struct is_not_same {
        typedef typename boost::mpl::not_< typename boost::is_same< T1, T2 >::type >::type type;
        BOOST_STATIC_CONSTANT(bool, value = (type::value));
    };
}
