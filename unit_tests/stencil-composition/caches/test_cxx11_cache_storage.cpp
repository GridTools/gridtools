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

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/offset_computation.hpp>

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

namespace test_multidimensional_caches {

    int test_ijcache() {

        typedef BACKEND::storage_traits_t::storage_info_t< 0, 6 > storage_info_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, storage_info_t > storage_t;
        typedef storage_wrapper< arg< 0, storage_t >,
            data_view< storage_t >,
            gridtools::tile< 0, 0, 0 >,
            gridtools::tile< 0, 0, 0 > > sw1_t;
        typedef detail::cache_impl< IJ, arg< 0, storage_t >, cache_io_policy::local, boost::mpl::void_ > cache_t;

        typedef cache_storage< cache_t, block_size< 8, 3, 1, 1, 1 >, extent< -1, 1, -2, 2, 0, 0, 0, 2, -1, 0 >, sw1_t >
            cache_storage_t;
        typedef accessor< 0, enumtype::in, extent<>, 6 > acc_t;

        typedef typename cache_storage_t::meta_t m_t;

        static constexpr m_t m_;
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 0 >() == 10, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 1 >() == 7, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 2 >() == 1, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 3 >() == 3, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 4 >() == 2, "error");

        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(1, 0, 0, 0, 0, 0)) == 1), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(2, 0, 0, 0, 0, 0)) == 2), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(3, 0, 0, 0, 0, 0)) == 3), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(4, 0, 0, 0, 0, 0)) == 4), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(5, 0, 0, 0, 0, 0)) == 5), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(6, 0, 0, 0, 0, 0)) == 6), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(7, 0, 0, 0, 0, 0)) == 7), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(8, 0, 0, 0, 0, 0)) == 8), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(9, 0, 0, 0, 0, 0)) == 9), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 0, 0, 0, 0)) == 0), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 1, 0, 0, 0, 0)) == 10), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 2, 0, 0, 0, 0)) == 20), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 3, 0, 0, 0, 0)) == 30), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 4, 0, 0, 0, 0)) == 40), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 5, 0, 0, 0, 0)) == 50), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 6, 0, 0, 0, 0)) == 60), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 1, 0, 0, 0)) == 70), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 2, 0, 0, 0)) == 140), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 0, 1, 0, 0)) == 70), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 0, 0, 1, 0)) == 210), "error");
        return true;
    }

    int test_kcache() {
        typedef BACKEND::storage_traits_t::storage_info_t< 0, 6 > storage_info_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, storage_info_t > storage_t;
        typedef storage_wrapper< arg< 0, storage_t >,
            data_view< storage_t >,
            gridtools::tile< 0, 0, 0 >,
            gridtools::tile< 0, 0, 0 > > sw1_t;
        typedef detail::cache_impl< K, arg< 0, storage_t >, cache_io_policy::local, boost::mpl::void_ > cache_t;

        typedef cache_storage< cache_t, block_size< 1, 1, 1, 1, 1 >, extent< 0, 0, 0, 0, -3, 2, 0, 1, 0, 3 >, sw1_t >
            cache_storage_t;
        typedef accessor< 0, enumtype::in, extent<>, 6 > acc_t;

        typedef typename cache_storage_t::meta_t m_t;

        static constexpr m_t m_;

        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 0 >() == 1, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 1 >() == 1, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 2 >() == 6, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 3 >() == 2, "error");
        GRIDTOOLS_STATIC_ASSERT(m_t::template dim< 4 >() == 4, "error");

        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, -3, 0, 0, 0)) == -3), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, -1, 0, 0, 0)) == -1), "error");
        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 2, 0, 0, 0)) == 2), "error");

        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 0, 1, 0, 0)) == 6), "error");

        GRIDTOOLS_STATIC_ASSERT(
            (_impl::get_cache_offset< typename cache_storage_t::meta_t >(acc_t(0, 0, 0, 0, 2, 0)) == 24), "error");

        return true;
    }

} // namespace test_multidimensional_caches

TEST(cache_storage, test_ij_multidim) { ASSERT_TRUE((test_multidimensional_caches::test_ijcache())); }

TEST(cache_storage, test_k_multidim) { ASSERT_TRUE((test_multidimensional_caches::test_kcache())); }
