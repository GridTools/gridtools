#pragma once
#include <boost/make_shared.hpp>

namespace gridtools {

    template < typename RedFunctor, enumtype::binop BinOp, typename ReductionType, typename... ExtraArgs >
    reduction_descriptor< ReductionType, BinOp,
        boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >
    make_reduction(const ReductionType initial_value, ExtraArgs...) {
        return reduction_descriptor< ReductionType, BinOp,
            boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >(
            initial_value);
    }

} // namespace gridtools
