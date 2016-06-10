#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include "stencil_composition/mss_metafunctions.hpp"

namespace gridtools {

/*!
   \fn esf_descriptor<ESF, ...> make_esf(plc1, plc2, plc3, ...)
   \brief Function to create a Elementary Stencil Function
   \param plc{i} placeholder which represents the i-th argument to the functor ESF

   Use this function to associate a stencil functor (stage) to
   arguments (actually, placeholders to arguments)
 */

/*!
   \fn mss_descriptor<...> make_esf(ExecutionEngine, esf1, esf2, ...)
   \brief Function to create a Multistage Stencil that can then be executed
   \param esf{i}  i-th Elementary Stencil Function created with make_esf or a list specified as independent ESFs created
   with make independent

   Use this function to create a multi-stage stencil computation
 */

/*!
   \fn independent_esf<...> make_independent(esf1, esf2, ...)
   \brief Function to create a list of independent Elementary Styencil Functions

   \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as
   independent

   Function to create a list of independent Elementary Styencil Functions. This is used to let the library compute tight
   bounds on blocks to be used by backends
 */

// clang-format off
#define _MAKE_ESF(z, n, nil)                                                                                           \
    template < typename ESF, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename A) >                                       \
        esf_descriptor< ESF,                                                                                           \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A) > >            \
        make_stage(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A)) {                                                           \
        return esf_descriptor< ESF,                                                                                    \
                   BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A) > > (); \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, _MAKE_ESF, _)

//clang-format on

#undef _MAKE_ESF

} // namespace gridtools
