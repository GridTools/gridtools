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

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include "common/generic_metafunctions/mpl_vector_flatten.hpp"
#include "stencil-composition/mss_metafunctions.hpp"
#include "stencil-composition/mss.hpp"

namespace gridtools {

// clang-format off
    /*!
       \fn mss_descriptor<...> make_esf(ExecutionEngine, esf1, esf2, ...)
       \brief Function to create a Multistage Stencil that can then be executed
       \param esf{i}  i-th Elementary Stencil Function created with make_esf or a list specified as independent ESFs created with make independent

       Use this function to create a multi-stage stencil computation
     */
#define _MAKE_MSS(z, ITN, nil)                                                                                \
    template < typename ExecutionEngine, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), typename EsfDescr) >         \
    mss_descriptor< ExecutionEngine,                                                                          \
        typename extract_mss_esfs< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(ITN)) <                      \
                                   BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > >::type,               \
        typename extract_mss_caches< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(ITN)) <                    \
                                     BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > > ::type >            \
            make_multistage(ExecutionEngine const &,                                                                 \
                BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(ITN), EsfDescr, const &BOOST_PP_INTERCEPT)) {        \
        return mss_descriptor< ExecutionEngine,                                                               \
                   typename extract_mss_esfs< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(ITN)) <           \
                                              BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > >::type,    \
               typename extract_mss_caches< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(ITN)) <             \
                                            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > >::type > (); \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, _MAKE_MSS, _)
#undef _MAKE_MSS

    /*!
       \fn independent_esf<...> make_independent(esf1, esf2, ...)
       \brief Function to create a list of independent Elementary Styencil Functions

       \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as independent

       Function to create a list of independent Elementary Styencil Functions. This is used to let the library compute tight bounds on blocks to be used by backends
     */
#define _MAKE_INDEPENDENT(z, ITN, nil)                                                                               \
    template < typename EsfDescr, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), typename EsfDescr) >                       \
        independent_esf< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(BOOST_PP_INC(ITN))) < EsfDescr,               \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > >                                                     \
        make_independent(                                                                                            \
            EsfDescr const &, BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(ITN), EsfDescr, const &BOOST_PP_INTERCEPT)) { \
        return independent_esf< BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(BOOST_PP_INC(ITN))) < EsfDescr,        \
                   BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(ITN), EsfDescr) > > ();                                          \
    }

    // clang-format on
    BOOST_PP_REPEAT(GT_MAX_INDEPENDENT, _MAKE_INDEPENDENT, _)
#undef _MAKE_INDEPENDENT
} // namespace gridtools
