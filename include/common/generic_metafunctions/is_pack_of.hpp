#pragma once
#include "accumulate.hpp"

namespace gridtools {
#ifdef CXX11_ENABLED

    /**
     * SFINAE for the case in which all the components of a parameter pack are of type determined by the predicate
     * Returns true also if the variadic pack is empty
     * Example of use:
     * template<typename ...Args, typename = is_pack_of<is_static_integral, Args...> >
     * void fn(Args... args) {}
     */
    template < template < typename > class Pred, typename... IntTypes >
    using is_pack_of =
        typename boost::enable_if_c< ((sizeof...(IntTypes) == 0) ||
                                         accumulate(logical_and(), true, Pred< IntTypes >::type::value...)),
            bool >::type;
#endif
}
