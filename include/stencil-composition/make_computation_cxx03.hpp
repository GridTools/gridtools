/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "conditionals/conditionals_macros_cxx03.hpp"
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "make_computation_helper_cxx03.hpp"

namespace gridtools {

#define _PAIR_(count, N, data) data##Type##N data##Value##N

#ifdef __CUDACC__
#define _POINTER_(z, n, nil) \
    computation< typename _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::reduction_type_t > *
#else
#define _POINTER_(z, n, nil) \
    boost::shared_ptr< computation< typename _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::reduction_type_t > >
#endif

#define _MAKE_COMPUTATION(z, n, nil)                                                                  \
    template < typename Backend,                                                                      \
        typename Domain,                                                                              \
        typename Grid,                                                                                \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                     \
    _POINTER_(z, n, nil)                                                                              \
    make_computation(Domain &domain, const Grid &grid, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) { \
        return make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(                           \
            domain, grid, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                           \
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION, _)
#undef _MAKE_COMPUTATION

/////////////////////////////////////////////////////////////////////////////////////
/// MAKE POSITIONAL COMPUTATIOS
////////////////////////////////////////////////////////////////////////////////////

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                                                       \
    template < typename Backend,                                                                                      \
        typename Domain,                                                                                              \
        typename Grid,                                                                                                \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                                     \
    _POINTER_(z, n, nil)                                                                                              \
    make_positional_computation(Domain &domain, const Grid &grid, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) {      \
        return make_computation_impl< true, Backend >(domain, grid, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue)); \
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_POSITIONAL_COMPUTATION, _)
#undef _MAKE_POSITIONAL_COMPUTATION
#undef _PAIR_
#undef _POINTER_
} // namespace gridtools
