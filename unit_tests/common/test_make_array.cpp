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
#include "common/defs.hpp"
#include "common/array.hpp"
#include "common/make_array.hpp"
#include "../test_helper.hpp"

using namespace gridtools;

TEST(make_array_test, only_int) {
    auto a = make_array(1, 2, 3);

    auto expected = array< int, 3 >{1, 2, 3};

    ASSERT_TYPE_EQ< decltype(expected), decltype(a) >();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, constexpr_only_int) {
    constexpr auto a = make_array(1, 2, 3);

    constexpr auto expected = array< int, 3 >{1, 2, 3};

    ASSERT_TYPE_EQ< decltype(expected), decltype(a) >();

    ASSERT_EQ(expected, a);
    constexpr bool force_constexpr = (expected == a);
    ASSERT_TRUE(force_constexpr);
}

TEST(make_array_test, int_and_long) {
    auto a = make_array(1, 2, 3l);

    auto expected = array< long int, 3 >{1l, 2l, 3l};

    ASSERT_TYPE_EQ< decltype(expected), decltype(a) >();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, int_and_double) {
    double a_double = 3;
    auto a = make_array(1, 2, a_double);

    auto expected = array< double, 3 >{1., 2., a_double};

    ASSERT_TYPE_EQ< decltype(expected), decltype(a) >();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, force_double_for_ints) {
    auto a = make_array< double >(1, 2, 3);

    auto expected = array< double, 3 >{1., 2., 3.};

    ASSERT_TYPE_EQ< decltype(expected), decltype(a) >();
    ASSERT_EQ(expected, a);
}
