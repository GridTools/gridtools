/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "../esf.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/make_shared.hpp>

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
