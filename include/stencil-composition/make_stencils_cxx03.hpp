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
