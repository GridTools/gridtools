#pragma once

// clang-format off
#if !defined(BOOST_MPL_PREPROCESSING_MODE)
#   include "boost/mpl/map/map50.hpp"
#endif

#include <boost/mpl/aux_/config/use_preprocessed.hpp>

#if !defined(BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS) \
    && !defined(BOOST_MPL_PREPROCESSING_MODE)
Error, use of preprocessed files is disabled
#else

#   include <boost/preprocessor/iterate.hpp>

namespace boost { namespace mpl {

#   define BOOST_PP_ITERATION_PARAMS_1 \
    (3,(51, 120, <boost/mpl/map/aux_/numbered.hpp>))
#   include BOOST_PP_ITERATE()

}}

#endif // BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
    // clang-format on
