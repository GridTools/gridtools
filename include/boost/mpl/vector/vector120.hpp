#ifndef BOOST_MPL_VECTOR_VECTOR120_HPP_INCLUDED
#define BOOST_MPL_VECTOR_VECTOR120_HPP_INCLUDED

// clang-format off
#if !defined(BOOST_MPL_PREPROCESSING_MODE)
#   include <boost/mpl/vector/vector50.hpp>
#endif

#include <boost/mpl/aux_/config/use_preprocessed.hpp>

#if !defined(BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS) \
    && !defined(BOOST_MPL_PREPROCESSING_MODE)
Error, use of preprocessed files is disabled
#else

#   include <boost/mpl/aux_/config/typeof.hpp>
#   include <boost/mpl/aux_/config/ctps.hpp>
#   include <boost/preprocessor/iterate.hpp>

namespace boost { namespace mpl {

#   define BOOST_PP_ITERATION_PARAMS_1 \
    (3,(51, 120, <boost/mpl/vector/aux_/numbered.hpp>))
#   include BOOST_PP_ITERATE()

}}

#endif // BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
// clang-format on

#endif // BOOST_MPL_VECTOR_VECTOR120_HPP_INCLUDED
