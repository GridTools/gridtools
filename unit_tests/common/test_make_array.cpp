/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "../test_helper.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/make_array.hpp>

using namespace gridtools;

TEST(make_array_test, only_int) {
    auto a = make_array(1, 2, 3);

    auto expected = array<int, 3>{1, 2, 3};

    ASSERT_TYPE_EQ<decltype(expected), decltype(a)>();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, int_and_long) {
    auto a = make_array(1, 2, 3l);

    auto expected = array<long int, 3>{1l, 2l, 3l};

    ASSERT_TYPE_EQ<decltype(expected), decltype(a)>();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, int_and_double) {
    double a_double = 3;
    auto a = make_array(1, 2, a_double);

    auto expected = array<double, 3>{1., 2., a_double};

    ASSERT_TYPE_EQ<decltype(expected), decltype(a)>();
    ASSERT_EQ(expected, a);
}

TEST(make_array_test, force_double_for_ints) {
    auto a = make_array<double>(1, 2, 3);

    auto expected = array<double, 3>{1., 2., 3.};

    ASSERT_TYPE_EQ<decltype(expected), decltype(a)>();
    ASSERT_EQ(expected, a);
}
