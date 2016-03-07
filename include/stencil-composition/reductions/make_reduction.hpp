#pragma once
#include <boost/make_shared.hpp>

namespace gridtools {

    template < typename RedFunctor, typename ReductionType, typename... ExtraArgs >
    reduction_descriptor< ReductionType,
        boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >
    make_reduction(ReductionType initial_value, ExtraArgs...) {
        return reduction_descriptor< ReductionType,
            boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >(
            initial_value);
    }

} // namespace gridtools
