/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/compose.hpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        TEST(compose, smoke) {
            auto testee =
                compose([](int x) { return x + 1; }, [](int x) { return 2 * x; }, [](int x, int y) { return x % y; });
            EXPECT_EQ(2 * (124 % 43) + 1, testee(124, 43));
        }
    } // namespace
} // namespace gridtools
