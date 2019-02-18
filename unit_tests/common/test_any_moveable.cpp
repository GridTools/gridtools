/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/any_moveable.hpp>

#include <gtest/gtest.h>
#include <memory>

namespace gridtools {

    TEST(any_moveable, smoke) {
        any_moveable x = 42;
        EXPECT_TRUE(x.has_value());
        EXPECT_EQ(typeid(int), x.type());
        EXPECT_EQ(42, any_cast<int>(x));
        auto &ref = any_cast<int &>(x);
        ref = 88;
        EXPECT_EQ(88, any_cast<int>(x));
        EXPECT_FALSE(any_cast<double *>(&x));
    }

    TEST(any_moveable, empty) { EXPECT_FALSE(any_moveable{}.has_value()); }

    TEST(any_moveable, move_only) {
        using testee_t = std::unique_ptr<int>;
        any_moveable x = testee_t(new int(42));
        EXPECT_EQ(42, *any_cast<testee_t const &>(x));
        any_moveable y = std::move(x);
        EXPECT_EQ(42, *any_cast<testee_t const &>(y));
    }
} // namespace gridtools
