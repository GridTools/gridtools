/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(array, test_copyctr) {
    constexpr array<uint_t, 4> a{4, 2, 3, 1};
    constexpr auto mod_a(a);
    ASSERT_TRUE((mod_a == array<uint_t, 4>{4, 2, 3, 1}));
    ASSERT_TRUE((mod_a[0] == 4));
}

TEST(array, iterate_empty) {
    array<uint_t, 0> a{};

    ASSERT_EQ(a.begin(), a.end());

    for (auto it = a.begin(); it < a.end(); ++it) {
        FAIL();
    }
}

#if __cplusplus >= 201402L
TEST(array, constexpr_compare) {
    constexpr array<uint_t, 3> a{0, 0, 0};
    constexpr array<uint_t, 3> b{0, 0, 0};
    constexpr array<uint_t, 3> c{0, 0, 1};

    constexpr bool eq = (a == b);
    constexpr bool neq = (a != c);

    ASSERT_TRUE(eq);
    ASSERT_TRUE(neq);
}
#endif

TEST(array, iterate) {
    const int N = 5;
    array<double, N> a{};

    ASSERT_EQ(N * sizeof(double), reinterpret_cast<char *>(a.end()) - reinterpret_cast<char *>(a.begin()));

    int count = 0;
    for (auto it = a.begin(); it < a.end(); ++it) {
        count++;
    }
    ASSERT_EQ(N, count);
}
