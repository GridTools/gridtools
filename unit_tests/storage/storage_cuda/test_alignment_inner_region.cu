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
#include <iostream>
#include <gtest/gtest.h>
#include <storage/storage-facility.hpp>
#include <common/defs.hpp>
#include <stencil-composition/storage_info_extender.hpp>

namespace gt = gridtools;

template < typename View, typename Ptr >
__global__ void check(View view, Ptr *pgres, gt::uint_t h1, gt::uint_t h2, gt::uint_t h3, gt::uint_t a) {
    *pgres = &view(h1, h2, h3);
}

template < typename Layout, gt::int_t I >
constexpr gt::uint_t add_or_not(gt::uint_t x) {
    return (Layout::find(1) == I) ? x : 0;
}

template < typename ValueType, gt::uint_t a, typename Layout >
void run() {
    ValueType **pgres;
    cudaMalloc(&pgres, sizeof(int));

    constexpr gt::uint_t h1 = 7;
    constexpr gt::uint_t h2 = 4;
    constexpr gt::uint_t h3 = 5;
    using info = gt::cuda_storage_info< 0, Layout, gt::halo< h1, h2, h3 >, gt::alignment< a > >;
    using store = gt::storage_traits< gt::enumtype::Cuda >::data_store_t< ValueType, info >;

    info i(1200, 1200, 12);
    store s(i);

    auto view = gt::make_device_view(s);

    ValueType *res;

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<< 1, 1 >>>(view, pgres, h1, h2, h3, a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast< std::uintptr_t >(res) % a, 0);

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<< 1, 1 >>>(view,
        pgres,
        h1 + add_or_not< Layout, 0 >(1),
        h2 + add_or_not< Layout, 1 >(1),
        h3 + add_or_not< Layout, 2 >(1),
        a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast< std::uintptr_t >(res) % a, 0);

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<< 1, 1 >>>(view,
        pgres,
        h1 + add_or_not< Layout, 0 >(2),
        h2 + add_or_not< Layout, 1 >(2),
        h3 + add_or_not< Layout, 2 >(2),
        a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast< std::uintptr_t >(res) % a, 0);

    cudaFree(pgres);
}

TEST(Storage, InnerRegionAlignmentCharCuda210) { run< char, 1024, gt::layout_map< 2, 1, 0 > >(); }

TEST(Storage, InnerRegionAlignmentIntCuda210) { run< int, 256, gt::layout_map< 2, 1, 0 > >(); }

TEST(Storage, InnerRegionAlignmentFloatCuda210) { run< float, 32, gt::layout_map< 2, 1, 0 > >(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda210) { run< double, 512, gt::layout_map< 2, 1, 0 > >(); }

TEST(Storage, InnerRegionAlignmentCharCuda012) { run< char, 1024, gt::layout_map< 0, 1, 2 > >(); }

TEST(Storage, InnerRegionAlignmentIntCuda012) { run< int, 256, gt::layout_map< 0, 1, 2 > >(); }

TEST(Storage, InnerRegionAlignmentFloatCuda012) { run< float, 32, gt::layout_map< 0, 1, 2 > >(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda012) { run< double, 512, gt::layout_map< 0, 1, 2 > >(); }

TEST(Storage, InnerRegionAlignmentCharCuda021) { run< char, 1024, gt::layout_map< 0, 2, 1 > >(); }

TEST(Storage, InnerRegionAlignmentIntCuda021) { run< int, 256, gt::layout_map< 0, 2, 1 > >(); }

TEST(Storage, InnerRegionAlignmentFloatCuda021) { run< float, 32, gt::layout_map< 0, 2, 1 > >(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda021) { run< double, 512, gt::layout_map< 0, 2, 1 > >(); }
