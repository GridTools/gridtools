/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
