#ifndef BOOST_PP_IS_ITERATING
/*=============================================================================
    Copyright (c) 2011 Eric Niebler
    Copyright (c) 2001-2011 Joel de Guzman

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#if !defined(BOOST_FUSION_VECTOR120_FWD_HPP_INCLUDED)
#define BOOST_FUSION_VECTOR120_FWD_HPP_INCLUDED

#include <boost/fusion/support/config.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

// clang-format off
#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
Error, use of preprocessed files is disabled
#else

namespace boost { namespace fusion
{
    // expand vector51 to vector120
    #define BOOST_PP_FILENAME_1 "boost/fusion/vector/vector120_fwd.hpp"
    #define BOOST_PP_ITERATION_LIMITS (51, 120)
    #include BOOST_PP_ITERATE()
}}

#endif // BOOST_FUSION_DONT_USE_PREPROCESSED_FILES

// clang-format on

#endif

#else

template < BOOST_PP_ENUM_PARAMS(BOOST_PP_ITERATION(), typename T) >
struct BOOST_PP_CAT(vector, BOOST_PP_ITERATION());

#endif
