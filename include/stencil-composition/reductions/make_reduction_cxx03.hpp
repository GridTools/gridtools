#pragma once
#include <boost/make_shared.hpp>
#include "../esf.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"

namespace gridtools {

#define _PARTMPL_(count, N, data) typename data##Type##N

#define _PARDECL_(count, N, data) data##Type##N data##Value##N
// clang-format off
#define _RED_DESCR(z, n, nil)                                                                               \
    template < typename RedFunctor,                                                                         \
        typename BinOp,                                                                                     \
        typename ReductionType,                                                                             \
        BOOST_PP_ENUM(BOOST_PP_INC(n), _PARTMPL_, Par) >                                                    \
        reduction_descriptor< ReductionType,                                                                \
            BinOp,                                                                                          \
            boost::mpl::vector1< esf_descriptor< RedFunctor,                                                \
                BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <                                         \
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), ParType) > > > >                                  \
        make_reduction(const ReductionType initial_value, BOOST_PP_ENUM(BOOST_PP_INC(n), _PARDECL_, Par)) { \
        return reduction_descriptor< ReductionType,                                                         \
                   BinOp,                                                                                   \
                   boost::mpl::vector1< esf_descriptor< RedFunctor,                                         \
                       BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <                                  \
                           BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), ParType) > > > > (initial_value);          \
    };
// clang-format on
    BOOST_PP_REPEAT(GT_MAX_MSS, _RED_DESCR, _)

#undef _RED_DESCR
#undef _PARDECL_
#undef _PARTMPL_

} // namespace gridtools
