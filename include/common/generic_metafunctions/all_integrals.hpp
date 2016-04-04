#pragma once
#include "accumulate.hpp"
namespace gridtools {
#ifdef CXX11_ENABLED

    /**
       SFINAE for the case in which all the components of a parameter pack are of integral type
    */
    template < typename... IntTypes >
    using all_integers =
        typename boost::enable_if_c< accumulate(logical_and(), boost::is_integral< IntTypes >::type::value...),
            bool >::type;

    /**
       SFINAE for the case in which all the components of a parameter pack are of static integral type
    */
    template < typename... IntTypes >
    using all_static_integers =
        typename boost::enable_if_c< accumulate(logical_and(), is_static_integral< IntTypes >::type::value...),
            bool >::type;

#endif
}
