/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/ldg_ptr.hpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        TEST(as_ldg_ptr, non_const_host) {
            float data[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

            auto ptr = as_ldg_ptr(&data[2]);
            EXPECT_EQ(*ptr, 2.0f);
            EXPECT_EQ(ptr + 2, as_ldg_ptr(&data[4]));
            EXPECT_EQ(ptr - 2, as_ldg_ptr(&data[0]));
            EXPECT_EQ(*(ptr + 2), 4.0f);
            EXPECT_EQ(*(ptr - 2), 0.0f);
            EXPECT_EQ(*(++ptr), 3.0f);
            EXPECT_EQ(*(ptr++), 3.0f);
            EXPECT_EQ(*(ptr--), 4.0f);
            EXPECT_EQ(*(--ptr), 2.0f);
            EXPECT_EQ((ptr + 2) - ptr, 2);
            *ptr = 5.0f;
            EXPECT_EQ(*ptr, 5.0f);
        }

        TEST(as_ldg_ptr, const_host) {
            float const data[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

            auto ptr = as_ldg_ptr(&data[2]);
            EXPECT_EQ(*ptr, 2.0f);
            EXPECT_EQ(ptr + 2, as_ldg_ptr(&data[4]));
            EXPECT_EQ(ptr - 2, as_ldg_ptr(&data[0]));
            EXPECT_EQ(*(ptr + 2), 4.0f);
            EXPECT_EQ(*(ptr - 2), 0.0f);
            EXPECT_EQ(*(++ptr), 3.0f);
            EXPECT_EQ(*(ptr++), 3.0f);
            EXPECT_EQ(*(ptr--), 4.0f);
            EXPECT_EQ(*(--ptr), 2.0f);
            EXPECT_EQ((ptr + 2) - ptr, 2);
        }
    } // namespace
} // namespace gridtools
