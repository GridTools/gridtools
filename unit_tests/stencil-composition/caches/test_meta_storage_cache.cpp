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

#include <gridtools/common/layout_map.hpp>
#include <gridtools/stencil-composition/caches/meta_storage_cache.hpp>

#include "../../test_helper.hpp"
#include "gtest/gtest.h"

using gridtools::layout_map;
using gridtools::meta_storage_cache;

TEST(meta_storage_cache, standard_layout) {
    constexpr int Dim0 = 2;
    constexpr int Dim1 = 3;
    constexpr int Dim2 = 4;

    using layout_t = layout_map<0, 1, 2>;
    using meta_t = meta_storage_cache<layout_t, 2, 3, 4>;
    constexpr meta_t meta;

    constexpr gridtools::uint_t expected_total_length = Dim0 * Dim1 * Dim2;
    ASSERT_STATIC_EQ(expected_total_length, meta_t::padded_total_length());

    EXPECT_EQ(Dim2 * Dim1, meta_t::stride<0>());
    EXPECT_EQ(Dim2, meta_t::stride<1>());
    EXPECT_EQ(1, meta_t::stride<2>());

    EXPECT_EQ(Dim0, meta_t::dim<0>());
    EXPECT_EQ(Dim1, meta_t::dim<1>());
    EXPECT_EQ(Dim2, meta_t::dim<2>());
}

TEST(meta_storage_cache, inverted_layout) {
    constexpr int Dim0 = 2;
    constexpr int Dim1 = 3;
    constexpr int Dim2 = 4;

    using layout_t = layout_map<2, 1, 0>;
    using meta_t = meta_storage_cache<layout_t, 2, 3, 4>;
    constexpr meta_t meta;

    constexpr gridtools::uint_t expected_total_length = Dim0 * Dim1 * Dim2;
    ASSERT_STATIC_EQ(expected_total_length, meta_t::padded_total_length());

    ASSERT_EQ(1, meta_t::stride<0>());
    ASSERT_EQ(Dim0, meta_t::stride<1>());
    ASSERT_EQ(Dim0 * Dim1, meta_t::stride<2>());

    ASSERT_EQ(Dim0, meta_t::dim<0>());
    ASSERT_EQ(Dim1, meta_t::dim<1>());
    ASSERT_EQ(Dim2, meta_t::dim<2>());
}
