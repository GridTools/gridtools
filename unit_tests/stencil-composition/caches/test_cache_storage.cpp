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

#include "backend_select.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil-composition/block_size.hpp>
#include <gridtools/stencil-composition/offset_computation.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

TEST(cache_storage, test_ij_multidim) {
    typedef backend_t::storage_traits_t::storage_info_t<0, 6> storage_info_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, storage_info_t> storage_t;
    typedef detail::cache_impl<IJ, arg<0, storage_t>, cache_io_policy::local, boost::mpl::void_> cache_t;

    typedef cache_storage<cache_t,
        block_size<8, 3, 1, 1, 1>,
        extent<-1, 1, -2, 2, 0, 0, 0, 2, -1, 0>,
        arg<0, storage_t>>
        cache_storage_t;
    typedef accessor<0, enumtype::in, extent<>, 6> acc_t;

    typedef typename cache_storage_t::meta_t m_t;

    static constexpr m_t m_;
#if !defined(__INTEL_COMPILER) || __INTEL_COMPILER != 1800
    // ICC 18 shows a strange bug here
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<0>() == 10, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<1>() == 7, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<2>() == 1, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<3>() == 3, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<4>() == 2, "error");
#endif

    ASSERT_EQ(1, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(1, 0, 0, 0, 0, 0)));
    ASSERT_EQ(2, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(2, 0, 0, 0, 0, 0)));
    ASSERT_EQ(3, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(3, 0, 0, 0, 0, 0)));
    ASSERT_EQ(4, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(4, 0, 0, 0, 0, 0)));
    ASSERT_EQ(5, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(5, 0, 0, 0, 0, 0)));
    ASSERT_EQ(6, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(6, 0, 0, 0, 0, 0)));
    ASSERT_EQ(7, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(7, 0, 0, 0, 0, 0)));
    ASSERT_EQ(8, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(8, 0, 0, 0, 0, 0)));
    ASSERT_EQ(9, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(9, 0, 0, 0, 0, 0)));
    ASSERT_EQ(0, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 0, 0, 0, 0)));
    ASSERT_EQ(10, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 1, 0, 0, 0, 0)));
    ASSERT_EQ(20, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 2, 0, 0, 0, 0)));
    ASSERT_EQ(30, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 3, 0, 0, 0, 0)));
    ASSERT_EQ(40, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 4, 0, 0, 0, 0)));
    ASSERT_EQ(50, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 5, 0, 0, 0, 0)));
    ASSERT_EQ(60, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 6, 0, 0, 0, 0)));
    ASSERT_EQ(70, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 1, 0, 0, 0)));
    ASSERT_EQ(140, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 2, 0, 0, 0)));
    ASSERT_EQ(70, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 0, 1, 0, 0)));
    ASSERT_EQ(210, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 0, 0, 1, 0)));
}

TEST(cache_storage, test_k_multidim) {
    typedef backend_t::storage_traits_t::storage_info_t<0, 6> storage_info_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, storage_info_t> storage_t;
    typedef detail::cache_impl<K, arg<0, storage_t>, cache_io_policy::local, boost::mpl::void_> cache_t;

    typedef cache_storage<cache_t, block_size<1, 1, 1, 1, 1>, extent<0, 0, 0, 0, -3, 2, 0, 1, 0, 3>, arg<0, storage_t>>
        cache_storage_t;
    typedef accessor<0, enumtype::in, extent<>, 6> acc_t;

    typedef typename cache_storage_t::meta_t m_t;

    static constexpr m_t m_;

#if !defined(__INTEL_COMPILER) || __INTEL_COMPILER != 1800
    // ICC 18 shows a strange bug here
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<0>() == 1, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<1>() == 1, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<2>() == 6, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<3>() == 2, "error");
    GRIDTOOLS_STATIC_ASSERT(m_t::template dim<4>() == 4, "error");
#endif

    ASSERT_EQ(-3, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, -3, 0, 0, 0)));
    ASSERT_EQ(-1, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, -1, 0, 0, 0)));
    ASSERT_EQ(2, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 2, 0, 0, 0)));
    ASSERT_EQ(6, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 0, 1, 0, 0)));
    ASSERT_EQ(24, compute_offset_cache<typename cache_storage_t::meta_t>(acc_t(0, 0, 0, 0, 2, 0)));
}
