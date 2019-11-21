/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/storage/storage_info.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace gridtools;
using testing::ElementsAre;

TEST(StorageInfo, Strides) {
    {
        storage_info<3> si(layout_map<0, 1, 2>(), 1, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(20, 5, 1));
        EXPECT_EQ(si.length(), 3 * 4 * 5);
    }
    {
        storage_info<3> si(layout_map<2, 0, 1>(), 1, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(1, 15, 3));
        EXPECT_EQ(si.length(), 3 * 4 * 5);
    }
    {
        storage_info<3> si(layout_map<-1, 0, 1>(), 1, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(0, 5, 1));
        EXPECT_EQ(si.length(), 4 * 5);
    }
}

TEST(StorageInfo, StridesAlignment) {
    {
        storage_info<3> si(layout_map<0, 1, 2>(), 32, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(128, 32, 1));
        EXPECT_GE(si.length(), 3 * 4 * 5);
        EXPECT_LE(si.length(), 3 * 4 * 32);
    }
    {
        storage_info<3> si(layout_map<2, 0, 1>(), 32, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(1, 32 * 5, 32));
        EXPECT_GE(si.length(), 3 * 4 * 5);
        EXPECT_LE(si.length(), 32 * 4 * 5);
    }
    {
        storage_info<3> si(layout_map<-1, 0, 1>(), 32, {3, 4, 5});
        EXPECT_THAT(si.strides(), ElementsAre(0, 32, 1));
        EXPECT_GE(si.length(), 4 * 5);
        EXPECT_LE(si.length(), 4 * 32);
    }
}

TEST(StorageInfo, IndexVariadic) {
    {
        storage_info<3> si(layout_map<0, 1, 2>(), 1, {3, 4, 5});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 1);
        EXPECT_EQ(si.index(0, 1, 0), 5);
        EXPECT_EQ(si.index(1, 0, 0), 20);
    }
    {
        storage_info<3> si(layout_map<2, 0, 1>(), 1, {3, 4, 5});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 3);
        EXPECT_EQ(si.index(0, 1, 0), 15);
        EXPECT_EQ(si.index(1, 0, 0), 1);
    }
    {
        storage_info<3> si(layout_map<-1, 0, 1>(), 1, {3, 4, 5});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 1);
        EXPECT_EQ(si.index(0, 1, 0), 5);
        EXPECT_EQ(si.index(1, 0, 0), 0);
        EXPECT_EQ(si.index(1, 1, 1), 6);
    }
}

TEST(StorageInfo, Simple) {
    {
        storage_info<3> si(layout_map<2, 1, 0>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 9);
        EXPECT_EQ(si.index(0, 0, 2), 18);

        EXPECT_EQ(si.index(0, 1, 0), 3);
        EXPECT_EQ(si.index(0, 1, 1), 12);
        EXPECT_EQ(si.index(0, 1, 2), 21);

        EXPECT_EQ(si.index(0, 2, 0), 6);
        EXPECT_EQ(si.index(0, 2, 1), 15);
        EXPECT_EQ(si.index(0, 2, 2), 24);

        EXPECT_EQ(si.index(1, 0, 0), 1);
        EXPECT_EQ(si.index(1, 0, 1), 10);
        EXPECT_EQ(si.index(1, 0, 2), 19);
    }
    {
        storage_info<3> si(layout_map<0, 1, 2>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 1);
        EXPECT_EQ(si.index(0, 0, 2), 2);

        EXPECT_EQ(si.index(0, 1, 0), 3);
        EXPECT_EQ(si.index(0, 1, 1), 4);
        EXPECT_EQ(si.index(0, 1, 2), 5);

        EXPECT_EQ(si.index(0, 2, 0), 6);
        EXPECT_EQ(si.index(0, 2, 1), 7);
        EXPECT_EQ(si.index(0, 2, 2), 8);

        EXPECT_EQ(si.index(1, 0, 0), 9);
        EXPECT_EQ(si.index(1, 0, 1), 10);
        EXPECT_EQ(si.index(1, 0, 2), 11);
    }
    {
        storage_info<3> si(layout_map<1, 0, 2>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index(0, 0, 0), 0);
        EXPECT_EQ(si.index(0, 0, 1), 1);
        EXPECT_EQ(si.index(0, 0, 2), 2);

        EXPECT_EQ(si.index(0, 1, 0), 9);
        EXPECT_EQ(si.index(0, 1, 1), 10);
        EXPECT_EQ(si.index(0, 1, 2), 11);

        EXPECT_EQ(si.index(0, 2, 0), 18);
        EXPECT_EQ(si.index(0, 2, 1), 19);
        EXPECT_EQ(si.index(0, 2, 2), 20);

        EXPECT_EQ(si.index(1, 0, 0), 3);
        EXPECT_EQ(si.index(1, 0, 1), 4);
        EXPECT_EQ(si.index(1, 0, 2), 5);
    }

    // test with different dims
    storage_info<4> x(layout_map<1, 2, 3, 0>(), 1, {5, 7, 8, 2});
    EXPECT_THAT(x.lengths(), ElementsAre(5, 7, 8, 2));
    EXPECT_THAT(x.strides(), ElementsAre(56, 8, 1, 280));
}

TEST(StorageInfo, ArrayAccess) {
    {
        storage_info<3> si(layout_map<2, 1, 0>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index({0, 0, 0}), 0);
        EXPECT_EQ(si.index({0, 0, 1}), 9);
        EXPECT_EQ(si.index({0, 0, 2}), 18);

        EXPECT_EQ(si.index({0, 1, 0}), 3);
        EXPECT_EQ(si.index({0, 1, 1}), 12);
        EXPECT_EQ(si.index({0, 1, 2}), 21);

        EXPECT_EQ(si.index({0, 2, 0}), 6);
        EXPECT_EQ(si.index({0, 2, 1}), 15);
        EXPECT_EQ(si.index({0, 2, 2}), 24);

        EXPECT_EQ(si.index({1, 0, 0}), 1);
        EXPECT_EQ(si.index({1, 0, 1}), 10);
        EXPECT_EQ(si.index({1, 0, 2}), 19);
    }
    {
        storage_info<3> si(layout_map<0, 1, 2>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index({0, 0, 0}), 0);
        EXPECT_EQ(si.index({0, 0, 1}), 1);
        EXPECT_EQ(si.index({0, 0, 2}), 2);

        EXPECT_EQ(si.index({0, 1, 0}), 3);
        EXPECT_EQ(si.index({0, 1, 1}), 4);
        EXPECT_EQ(si.index({0, 1, 2}), 5);

        EXPECT_EQ(si.index({0, 2, 0}), 6);
        EXPECT_EQ(si.index({0, 2, 1}), 7);
        EXPECT_EQ(si.index({0, 2, 2}), 8);

        EXPECT_EQ(si.index({1, 0, 0}), 9);
        EXPECT_EQ(si.index({1, 0, 1}), 10);
        EXPECT_EQ(si.index({1, 0, 2}), 11);
    }
    {
        storage_info<3> si(layout_map<1, 0, 2>(), 1, {3, 3, 3});
        EXPECT_EQ(si.index({0, 0, 0}), 0);
        EXPECT_EQ(si.index({0, 0, 1}), 1);
        EXPECT_EQ(si.index({0, 0, 2}), 2);

        EXPECT_EQ(si.index({0, 1, 0}), 9);
        EXPECT_EQ(si.index({0, 1, 1}), 10);
        EXPECT_EQ(si.index({0, 1, 2}), 11);

        EXPECT_EQ(si.index({0, 2, 0}), 18);
        EXPECT_EQ(si.index({0, 2, 1}), 19);
        EXPECT_EQ(si.index({0, 2, 2}), 20);

        EXPECT_EQ(si.index({1, 0, 0}), 3);
        EXPECT_EQ(si.index({1, 0, 1}), 4);
        EXPECT_EQ(si.index({1, 0, 2}), 5);
    }
}

TEST(StorageInfo, Alignment) {
    {
        storage_info<4> x(layout_map<1, 2, 3, 0>(), 32, {5, 7, 32, 2});
        EXPECT_THAT(x.lengths(), ElementsAre(5, 7, 32, 2));
        EXPECT_THAT(x.strides(), ElementsAre(32 * 7, 32, 1, 5 * 32 * 7));
    }
    {
        storage_info<4> x(layout_map<1, 2, 3, 0>(), 32, {7, 11, 3, 10});
        EXPECT_THAT(x.lengths(), ElementsAre(7, 11, 3, 10));
        EXPECT_THAT(x.strides(), ElementsAre(32 * 11, 32, 1, 32 * 11 * 7));
        EXPECT_EQ(x.index(0, 0, 0, 0), 0);
        EXPECT_EQ(x.index(0, 0, 1, 0), 1);
        EXPECT_EQ(x.index(0, 0, 2, 0), 2);
    }
    {
        storage_info<4> x(layout_map<3, 2, 1, 0>(), 32, {3, 11, 14, 10});
        EXPECT_THAT(x.lengths(), ElementsAre(3, 11, 14, 10));
        EXPECT_THAT(x.strides(), ElementsAre(1, 32, 32 * 11, 32 * 11 * 14));
        EXPECT_EQ(x.index(0, 0, 0, 0), 0);
        EXPECT_EQ(x.index(0, 1, 0, 0), 32);
        EXPECT_EQ(x.index(0, 0, 1, 0), 32 * 11);
        EXPECT_EQ(x.index(0, 0, 0, 1), 32 * 11 * 14);
    }
    {
        storage_info<4> x(layout_map<1, -1, -1, 0>(), 32, {7, 7, 8, 10});
        EXPECT_THAT(x.lengths(), ElementsAre(7, 7, 8, 10));
        EXPECT_THAT(x.strides(), ElementsAre(1, 0, 0, 32));
        EXPECT_EQ(x.index(0, 0, 0, 0), 0);
        EXPECT_EQ(x.index(0, 1, 0, 0), 0);
        EXPECT_EQ(x.index(0, 0, 1, 0), 0);
        EXPECT_EQ(x.index(0, 0, 0, 1), 32);
        EXPECT_GE(x.length(), 7 * 10);
        EXPECT_LE(x.length(), 32 * 10);
    }
}

TEST(StorageInfo, Equal) {
    storage_info<3> si1(layout_map<0, 1, 2>(), 16, {9, 11, 13});
    storage_info<3> si2(layout_map<0, 1, 2>(), 16, {9, 11, 13});
    EXPECT_EQ(si1, si2);
}

TEST(StorageInfo, SizesNotEqual) {
    storage_info<3> si1(layout_map<0, 1, 2>(), 16, {9, 11, 13});
    storage_info<3> si2(layout_map<0, 1, 2>(), 16, {9, 11, 15});
    EXPECT_NE(si1, si2);
}
