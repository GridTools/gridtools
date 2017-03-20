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
#pragma once
#include "../computation.hpp"
#include "fill_conditionals_cxx03.hpp"
#include "if_.hpp"
#include "../make_computation_helper_cxx03.hpp"

namespace gridtools {

#ifdef __CUDACC__
#define _POINTER_(z, n, nil) \
    computation< typename _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::reduction_type_t > *
#define _MAKE_POINTER_(data) new data
#else
#define _POINTER_(z, n, nil) \
    boost::shared_ptr< computation< typename _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::reduction_type_t > >
#define _MAKE_POINTER_(data) boost::make_shared< data >
#endif

#define _PAIR_(count, N, data) data##Type##N data##Value##N

#define _MAKE_COMPUTATION_IMPL(z, n, nil)                                                                           \
    template < bool Positional,                                                                                     \
        typename Backend,                                                                                           \
        typename Domain,                                                                                            \
        typename Grid,                                                                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                                   \
    _POINTER_(z, n, nil)                                                                                            \
    make_computation_impl(Domain &domain, const Grid &grid, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) {          \
        typedef typename boost::mpl::fold< boost::mpl::vector< BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) >,    \
            boost::mpl::vector0<>,                                                                                  \
            boost::mpl::if_< is_condition< boost::mpl::_2 >,                                                        \
                                               construct_conditionals_set< boost::mpl::_1, boost::mpl::_2 >,        \
                                               boost::mpl::_1 > >::type conditionals_set_mpl_t;                     \
                                                                                                                    \
        typedef typename boost::mpl::fold< conditionals_set_mpl_t,                                                  \
            boost::mpl::set0<>,                                                                                     \
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type conditionals_check_t;                      \
                                                                                                                    \
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< conditionals_check_t >::type::value ==                           \
                                    boost::mpl::size< conditionals_set_mpl_t >::type::value),                       \
            "Either you yoused the same switch_variable (or conditional) twice, "                                   \
            "or you used in the same computation two or more switch_variable (or conditional)"                      \
            "with the same index. The index Id in condition_variable<Type, Id> (or conditional<Id>)"                \
            "must be unique to the computation, and can be used only in one switch_ statement.");                   \
                                                                                                                    \
        typedef typename boost::fusion::result_of::as_set< conditionals_set_mpl_t >::type conditionals_set_t;       \
        conditionals_set_t conditionals_set_;                                                                       \
                                                                                                                    \
        fill_conditionals(conditionals_set_, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                      \
                                                                                                                    \
        typedef intermediate< Backend,                                                                              \
            meta_array< typename BOOST_PP_CAT(meta_array_generator, BOOST_PP_INC(n)) < boost::mpl::vector0<>,       \
                                  BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) >::type,                           \
            boost::mpl::quote1< is_computation_token > >,                                                           \
            Domain, Grid, conditionals_set_t,                                                                       \
            typename _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::reduction_type_t,                         \
            Positional > intermediate_t;                                                                            \
        return _MAKE_POINTER_(intermediate_t)(boost::ref(domain),                                                   \
            grid,                                                                                                   \
            conditionals_set_,                                                                                      \
            _impl::reduction_helper< BOOST_PP_CAT(MssType, n) >::extract_initial_value(BOOST_PP_CAT(MssValue, n))); \
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION_IMPL, _)

#undef _MAKE_COMPUTATION_IMPL
#undef _PAIR_
#undef _POINTER_
#undef _MAKE_POINTER_

} // namespace gridtools
