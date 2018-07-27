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
#include "../test_helper.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/tuple.hpp>
#include <string>

using namespace gridtools;

TEST(tuple, basic_test) {
    tuple<int, float> t1(1, 0.5f);

    ASSERT_EQ(t1.size(), 2);
    EXPECT_EQ(get<0>(t1), 1);
    EXPECT_EQ(get<1>(t1), 0.5f);

    tuple<int, float> t2;
    ASSERT_EQ(t2.size(), 2);
    get<0>(t2) = 2;
    get<1>(t2) = 1.0f;

    EXPECT_EQ(get<0>(t2), 2);
    EXPECT_EQ(get<1>(t2), 1.0f);
}

TEST(tuple, make_tuple_test) {
    auto t = make_tuple(0.5, 1, 0.5f);
    ASSERT_EQ(t.size(), 3);
    ASSERT_TYPE_EQ<decltype(t), tuple<double, int, float>>();

    EXPECT_EQ(get<0>(t), 0.5);
    EXPECT_EQ(get<1>(t), 1);
    EXPECT_EQ(get<2>(t), 0.5f);
}

TEST(tuple, constexpr_test) {
    constexpr tuple<float, int> t(0.5f, 3);
    constexpr auto t0 = get<0>(t);
    constexpr auto t1 = get<1>(t);
    GRIDTOOLS_STATIC_ASSERT(t.size() == 2, "");
    GRIDTOOLS_STATIC_ASSERT(t0 == 0.5f, "");
    GRIDTOOLS_STATIC_ASSERT(t1 == 3, "");
}

TEST(tuple, swap_test) {
    auto t1 = gridtools::make_tuple(0.0, 1, std::string("2"));
    auto t2 = gridtools::make_tuple(3.0, 4, std::string("5"));

    ASSERT_EQ(get<0>(t1), 0.0);
    ASSERT_EQ(get<1>(t1), 1);
    ASSERT_EQ(get<2>(t1), std::string("2"));
    ASSERT_EQ(get<0>(t2), 3.0);
    ASSERT_EQ(get<1>(t2), 4);
    ASSERT_EQ(get<2>(t2), std::string("5"));

    swap(t1, t2);

    ASSERT_EQ(get<0>(t1), 3.0);
    ASSERT_EQ(get<1>(t1), 4);
    ASSERT_EQ(get<2>(t1), std::string("5"));
    ASSERT_EQ(get<0>(t2), 0.0);
    ASSERT_EQ(get<1>(t2), 1);
    ASSERT_EQ(get<2>(t2), std::string("2"));
}
