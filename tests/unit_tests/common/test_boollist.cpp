/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/boollist.hpp>

#include <cstdlib>

#include <gtest/gtest.h>

#include <gridtools/common/layout_map.hpp>

using namespace gridtools;

TEST(boollist, functional) {
    for (int i = 0; i < 100000; ++i) {
        bool v0 = std::rand() % 2, v1 = std::rand() % 2, v2 = std::rand() % 2;

        boollist<3> bl1(v0, v1, v2);

        EXPECT_EQ(bl1.value(0), v0);
        EXPECT_EQ(bl1.value(1), v1);
        EXPECT_EQ(bl1.value(2), v2);

        boollist<3> bl2 = bl1.permute<gridtools::layout_map<1, 2, 0>>();

        EXPECT_EQ(bl2.value(0), v2);
        EXPECT_EQ(bl2.value(1), v0);
        EXPECT_EQ(bl2.value(2), v1);

        boollist<3> bl3 = bl1.permute<gridtools::layout_map<2, 1, 0>>();

        EXPECT_EQ(bl3.value(0), v2);
        EXPECT_EQ(bl3.value(1), v1);
        EXPECT_EQ(bl3.value(2), v0);

        boollist<3> bl4 = bl1.permute<gridtools::layout_map<0, 1, 2>>();

        EXPECT_EQ(bl4.value(0), v0);
        EXPECT_EQ(bl4.value(1), v1);
        EXPECT_EQ(bl4.value(2), v2);
    }
}
