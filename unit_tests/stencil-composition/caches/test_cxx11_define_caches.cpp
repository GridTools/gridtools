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
/**
   @file
   @brief File containing tests for the define_cache construct
*/

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <common/gt_assert.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/caches/define_caches.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#elif defined(__AVX512F__)
#define BACKEND backend< Mic, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

TEST(define_caches, test_sequence_caches) {
    typedef BACKEND::storage_traits_t::storage_info_t< 0, 3 > storage_info_t;
    typedef BACKEND::storage_traits_t::data_store_t< float_type, storage_info_t > storage_t;

    typedef gridtools::arg< 0, storage_t > arg0_t;
    typedef gridtools::arg< 1, storage_t > arg1_t;
    typedef gridtools::arg< 2, storage_t > arg2_t;

    typedef decltype(gridtools::define_caches(cache< IJ, cache_io_policy::fill >(arg0_t()),
        cache< IJK, cache_io_policy::flush >(arg1_t()),
        cache< K, cache_io_policy::local >(arg2_t()))) cache_sequence_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< cache_sequence_t,
            boost::mpl::vector3< detail::cache_impl< IJ, arg0_t, cache_io_policy::fill, boost::mpl::void_ >,
                                detail::cache_impl< IJK, arg1_t, cache_io_policy::flush, boost::mpl::void_ >,
                                detail::cache_impl< K, arg2_t, cache_io_policy::local, boost::mpl::void_ > > >::value),
        "Failed TEST");

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > interval_;

    typedef decltype(
        gridtools::cache< K, cache_io_policy::flush, interval_ >(arg0_t(), arg1_t())) caches_ret_sequence_4_t;
    typedef decltype(
        gridtools::cache< IJ, cache_io_policy::fill >(arg0_t(), arg1_t(), arg2_t())) caches_ret_sequence_3_t;
    typedef decltype(gridtools::cache< IJK, cache_io_policy::fill >(arg0_t(), arg1_t())) caches_ret_sequence_2_t;
    typedef decltype(gridtools::cache< IJ, cache_io_policy::fill >(arg0_t())) caches_ret_sequence_1_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< caches_ret_sequence_4_t,
            boost::mpl::vector2< detail::cache_impl< K, arg0_t, cache_io_policy::flush, interval_ >,
                                detail::cache_impl< K, arg1_t, cache_io_policy::flush, interval_ > > >::value),
        "Failed TEST");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< caches_ret_sequence_3_t,
            boost::mpl::vector3< detail::cache_impl< IJ, arg0_t, cache_io_policy::fill, boost::mpl::void_ >,
                                detail::cache_impl< IJ, arg1_t, cache_io_policy::fill, boost::mpl::void_ >,
                                detail::cache_impl< IJ, arg2_t, cache_io_policy::fill, boost::mpl::void_ > > >::value),
        "Failed TEST");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< caches_ret_sequence_2_t,
            boost::mpl::vector2< detail::cache_impl< IJK, arg0_t, cache_io_policy::fill, boost::mpl::void_ >,
                                detail::cache_impl< IJK, arg1_t, cache_io_policy::fill, boost::mpl::void_ > > >::value),
        "Failed TEST");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< caches_ret_sequence_1_t,
            boost::mpl::vector1< detail::cache_impl< IJ, arg0_t, cache_io_policy::fill, boost::mpl::void_ > > >::value),
        "Failed TEST");

    ASSERT_TRUE(true);
}
