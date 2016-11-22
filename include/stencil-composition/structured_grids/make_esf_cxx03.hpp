#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include "stencil-composition/mss_metafunctions.hpp"

namespace gridtools {

    /*!
       \fn esf_descriptor<ESF, ...> make_esf(plc1, plc2, plc3, ...)
       \brief Function to create a Elementary Stencil Function
       \param plc{i} placeholder which represents the i-th argument to the functor ESF

       Use this function to associate a stencil functor (stage) to
       arguments (actually, placeholders to arguments)
     */

#define _MAKE_ESF(z, n, nil)                                            \
    template <typename ESF,                                             \
              BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename A)>        \
    esf_descriptor<ESF, BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A)> > \
    make_esf(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A)) {                \
        return esf_descriptor<ESF, BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), A)> >(); \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, _MAKE_ESF, _)
#undef _MAKE_ESF

} // namespace gridtools
