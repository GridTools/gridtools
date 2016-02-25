#pragma once
#include <boost/make_shared.hpp>

namespace gridtools {

    template < typename RedFunctor, typename... ExtraArgs >
    mss_descriptor< enumtype::execute< enumtype::forward >,
        boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > >,
        true > make_reduction(ExtraArgs...) {
        return mss_descriptor< enumtype::execute< enumtype::forward >,
            boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > >,
            true >();
    }

} // namespace gridtools
