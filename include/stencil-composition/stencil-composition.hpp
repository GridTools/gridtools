/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/cat.hpp>

// clang-format off
#ifdef GT_VECTOR_LIMIT_SIZE
  #if GT_VECTOR_LIMIT_SIZE > 20 && GT_VECTOR_LIMIT_SIZE < 51
    #undef FUSION_MAX_VECTOR_SIZE
    #undef FUSION_MAX_MAP_SIZE
    #define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
    #define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
    #define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
    #define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
  #elif GT_VECTOR_LIMIT_SIZE > 50
    #define BOOST_FUSION_DONT_USE_PREPROCESSED_FILES
    #undef FUSION_MAX_MAP_SIZE
    #undef FUSION_MAX_VECTOR_SIZE
    #define FUSION_MAX_MAP_SIZE GT_VECTOR_LIMIT_SIZE
    #define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE

    #define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

    #define AUXBOOST_VECTOR_HEADER BOOST_PP_CAT(vector, GT_VECTOR_LIMIT_SIZE).hpp /**/

    #include BOOST_PP_STRINGIZE(boost/mpl/AUXBOOST_VECTOR_HEADER)

    #include BOOST_PP_STRINGIZE(boost/fusion/AUXBOOST_VECTOR_HEADER)

    #include <boost/mpl/vector.hpp>
    #include <boost/mpl/vector_c.hpp>

    #undef AUXBOOST_VECTOR_HEADER

    #define AUXBOOST_MAP_HEADER BOOST_PP_CAT(map, GT_VECTOR_LIMIT_SIZE).hpp /**/

    #include BOOST_PP_STRINGIZE(boost/fusion/AUXBOOST_MAP_HEADER)
    #undef AUXBOOST_MAP_HEADER

    #include <boost/fusion/container/vector.hpp>

  #endif
#endif
// clang-format on

//  #include <common/fusion/map120.hpp>

//#define BOOST_MPL_PREPROCESSING_MODE
//#include <boost/mpl/vector.hpp>
//#undef BOOST_MPL_PREPROCESSING_MODE

//#include <boost/mpl/vector_c.hpp>

//#include <common/fusion/vector120.hpp>

//#include <boost/fusion/container/vector.hpp>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "esf.hpp"
#include "intermediate_impl.hpp"
#include "make_computation.hpp"
#include "make_stage.hpp"
#include "make_stencils.hpp"
#include "../storage/storage-facility.hpp"
#ifdef CXX11_ENABLED
#include "expandable_parameters/make_computation_expandable.hpp"
#else
#include "make_computation.hpp"
#endif
#include "grid.hpp"
#include "grid_traits.hpp"
#include "stencil.hpp"
#include "storage_info_extender.hpp"

#ifndef STRUCTURED_GRIDS
#include "icosahedral_grids/grid.hpp"
#endif
