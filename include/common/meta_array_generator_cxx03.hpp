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

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

namespace gridtools {

#define _META_ARRAY_GENERATOR_(z, n, nil)                                                                             \
    template < typename Vector, typename First, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >             \
    struct BOOST_PP_CAT(meta_array_generator, BOOST_PP_INC(BOOST_PP_INC(n))) {                                        \
        typedef typename BOOST_PP_CAT(                                                                                \
            meta_array_generator, BOOST_PP_INC(n))< typename boost::mpl::push_back< Vector, First >::type,            \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) >::type type;                                              \
    };                                                                                                                \
                                                                                                                      \
    template < typename Vector,                                                                                       \
        typename Mss1,                                                                                                \
        typename Mss2,                                                                                                \
        typename Cond,                                                                                                \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                                     \
    struct BOOST_PP_CAT(meta_array_generator,                                                                         \
        BOOST_PP_INC(BOOST_PP_INC(                                                                                    \
            n)))< Vector, condition< Mss1, Mss2, Cond >, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) > {           \
        typedef condition< typename BOOST_PP_CAT(meta_array_generator, BOOST_PP_INC(BOOST_PP_INC(n))) < Vector,       \
            Mss1,                                                                                                     \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) >::type,                                                   \
            typename BOOST_PP_CAT(meta_array_generator,                                                               \
                BOOST_PP_INC(BOOST_PP_INC(n)))< Vector, Mss2, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) >::type, \
            Cond > type;                                                                                              \
    };

    template < typename Mss1, typename Mss2, typename Cond >
    struct condition;

    template < typename Vector >
    struct meta_array_generator0 {
        typedef Vector type;
    };

    template < typename Vector, typename Mss1 >
    struct meta_array_generator1 {
        typedef typename boost::mpl::push_back< Vector, Mss1 >::type type;
    };

    template < typename Vector, typename Mss1, typename Mss2, typename Cond >
    struct meta_array_generator1< Vector, condition< Mss1, Mss2, Cond > > {
        typedef condition< typename meta_array_generator1< Vector, Mss1 >::type,
            typename meta_array_generator1< Vector, Mss2 >::type,
            Cond > type;
    };

    BOOST_PP_REPEAT(GT_MAX_MSS, _META_ARRAY_GENERATOR_, _)

} // namespace gridtools
