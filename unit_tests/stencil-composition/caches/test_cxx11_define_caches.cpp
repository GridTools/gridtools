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
/**
   @file
   @brief File containing tests for the define_cache construct
*/

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/caches/define_caches.hpp"

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
  #ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
  #else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
  #endif
#endif

TEST(define_caches, test_sequence_caches)
{
#ifdef __CUDACC__
    typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
#else
    typedef gridtools::layout_map<0,1,2> layout_t;//stride 1 on k
#endif
    typedef gridtools::BACKEND::storage_type<float_type, gridtools::BACKEND::storage_info<0,layout_t> >::type storage_type;

    typedef gridtools::arg<0,storage_type> arg0_t;
    typedef gridtools::arg<1,storage_type> arg1_t;
    typedef gridtools::arg<2,storage_type> arg2_t;

    typedef decltype(gridtools::define_caches(
        cache< IJ, fill >(arg0_t()), cache< IJK, flush >(arg1_t()), cache< K, local >(arg2_t()))) cache_sequence_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal< cache_sequence_t,
                                boost::mpl::vector3< detail::cache_impl< IJ, arg0_t, fill >,
                                                    detail::cache_impl< IJK, arg1_t, flush >,
                                                    detail::cache_impl< K, arg2_t, local > > >::value),
        "Failed TEST");

    typedef decltype(gridtools::cache< IJ, fill >(arg0_t(), arg1_t(), arg2_t())) caches_ret_sequence_3_t;
    typedef decltype(gridtools::cache< IJK, fill >(arg0_t(), arg1_t())) caches_ret_sequence_2_t;
    typedef decltype(gridtools::cache< IJ, fill >(arg0_t())) caches_ret_sequence_1_t;

    static_assert((boost::mpl::equal< caches_ret_sequence_3_t,
                      boost::mpl::vector3< detail::cache_impl< IJ, arg0_t, fill >,
                                          detail::cache_impl< IJ, arg1_t, fill >,
                                          detail::cache_impl< IJ, arg2_t, fill > > >::value),
        "Failed TEST");
    static_assert((boost::mpl::equal< caches_ret_sequence_2_t,
                      boost::mpl::vector2< detail::cache_impl< IJK, arg0_t, fill >,
                                          detail::cache_impl< IJK, arg1_t, fill > > >::value),
        "Failed TEST");
    static_assert((boost::mpl::equal< caches_ret_sequence_1_t,
                      boost::mpl::vector1< detail::cache_impl< IJ, arg0_t, fill > > >::value),
        "Failed TEST");

    ASSERT_TRUE(true);
}
