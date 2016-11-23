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
#include "gtest/gtest.h"
#include <common/defs.hpp>
#include <storage/storage-facility.hpp>

using namespace gridtools;

TEST(storage_info, test_component) {
    typedef layout_map< 0, 1, 2 > layout;
#ifdef CXX11_ENABLED
    typedef storage_traits< enumtype::Host >::meta_storage_type< 0, layout > meta_data_t;
    typedef storage_traits< enumtype::Host >::storage_type< float, meta_data_t > storage_t;
#else
    typedef storage_traits< enumtype::Host >::meta_storage_type< 0, layout >::type meta_data_t;
    typedef storage_traits< enumtype::Host >::storage_type< float, meta_data_t >::type storage_t;
#endif
    meta_data_t meta_obj(10, 10, 10);
    storage_t st_obj(meta_obj, "in");
}

TEST(storage_info, test_equality) {
    typedef gridtools::layout_map< 0, 1, 2 > layout_t1;
    typedef gridtools::meta_storage_base< static_int< 0 >, layout_t1, false > meta_t1;
    meta_t1 m0(11, 12, 13);
    meta_t1 m1(11, 12, 13);
    meta_t1 m2(12, 123, 13);
    ASSERT_TRUE((m0 == m1) && "storage info equality test failed!");
    ASSERT_TRUE((m1 == m0) && "storage info equality test failed!");
    ASSERT_TRUE(!(m2 == m0) && "storage info equality test failed!");
}

TEST(storage_info, test_interface) {

#if defined(CUDA8)

    typedef gridtools::layout_map< 0, 1, 2, 3, 4 > layout_t;
    typedef typename gridtools::meta_storage_base< static_int< 0 >, layout_t, false > meta_t;

    constexpr meta_t meta_{11u, 12u, 13u, 14u, 15u};

    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 2 >() == 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 3 >() == 14), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides(4) == 15), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(3) == 15 * 14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(2) == 15 * 14 * 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(1) == 15 * 14 * 13 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(0) == 15 * 14 * 13 * 12 * 11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_t::strides< 4 >(meta_.strides()) == 1), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides< 4 >() == 1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides< 3 >() == 15), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides< 2 >() == 15 * 14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides< 1 >() == 15 * 14 * 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides< 0 >() == 15 * 14 * 13 * 12), "error");

    // aligned meta_storage test cases
    using halo_t = gridtools::halo< 0, 0, 0 >;
    using align_t = gridtools::aligned< 32 >;

#ifndef __CUDACC__
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< static_int< 0 >, gridtools::layout_map< 0, 1, 2 >, false >,
        align_t,
        halo_t >
        meta_aligned_1{11, 12, 13};
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< static_int< 0 >, gridtools::layout_map< 0, 2, 1 >, false >,
        align_t,
        halo_t >
        meta_aligned_2{11, 12, 13};
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< static_int< 0 >, gridtools::layout_map< 2, 1, 0 >, false >,
        align_t,
        halo_t >
        meta_aligned_3{11, 12, 13};

    // check unaligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 2 >() == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(1) == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 2 >() == 13), "error");

    // check aligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 2 >() == 32), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(1) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 0 >() == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 2 >() == 13), "error");

    // check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(2) == 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(1) == 13 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(0) == 13 * 12 * 11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 2 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 1 >() == 1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 0 >() == 12 * 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(2) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(1) == 11 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(0) == 11 * 12 * 13), "error");

    // check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(2) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(1) == 32 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(0) == 32 * 12 * 11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 2 >() == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 1 >() == 1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 0 >() == 32 * 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(2) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(1) == 32 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(0) == 32 * 12 * 13), "error");
#endif
#else // CUDA8

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    gridtools::meta_storage_base< static_int< 0 >, layout_t, false > meta_(11, 12, 13);
    ASSERT_TRUE((meta_.dim< 0 >() == 11));
    ASSERT_TRUE((meta_.dim< 1 >() == 12));
    ASSERT_TRUE((meta_.dim< 2 >() == 13));

    ASSERT_TRUE((meta_.strides(2) == 13));
    ASSERT_TRUE((meta_.strides(1) == 13 * 12));
    ASSERT_TRUE((meta_.strides(0) == 13 * 12 * 11));

    ASSERT_TRUE((meta_.strides< 2 >(meta_.strides()) == 1));
    ASSERT_TRUE((meta_.strides< 1 >(meta_.strides()) == 13));
    ASSERT_TRUE((meta_.strides< 0 >(meta_.strides()) == 13 * 12));
#endif

#ifdef CXX11_ENABLED // this checks are performed in cxx11 mode
    // create simple aligned meta storage
    typedef gridtools::halo< 0, 0, 0 > halo_t1;
    typedef gridtools::aligned< 32 > align_t1;
    gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< static_int< 0 >, gridtools::layout_map< 0, 1, 2 >, false >,
        align_t1,
        halo_t1 >
        meta_aligned_1nc(11, 12, 13);
    // check if unaligned dims and strides are correct
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(0) == 11) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(1) == 12) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(2) == 13) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(2) == 13) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(1) == 13 * 12) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(0) == 13 * 12 * 11) && "error");
    // create a storage and pass meta_data
    gridtools::storage< gridtools::base_storage< gridtools::wrap_pointer< float >, decltype(meta_aligned_1nc), 1 > >
        storage(meta_aligned_1nc, -1.0f);
    ASSERT_TRUE((storage.meta_data().unaligned_dim(0) == 11) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_dim(1) == 12) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_dim(2) == 13) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(2) == 13) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(1) == 13 * 12) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(0) == 13 * 12 * 11) && "error");
#endif // CXX11_ENABLED
}
