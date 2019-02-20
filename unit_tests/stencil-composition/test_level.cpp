/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../test_helper.hpp"
#include "gtest/gtest.h"

#include <gridtools/stencil-composition/level.hpp>

using namespace gridtools;

TEST(test_level, level_to_index) {
    ASSERT_TYPE_EQ<typename level_to_index<level<0, -2, 2>>::type, level_index<0, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, -1, 2>>::type, level_index<1, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, 1, 2>>::type, level_index<2, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, 2, 2>>::type, level_index<3, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, -2, 2>>::type, level_index<4, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, -1, 2>>::type, level_index<5, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, 1, 2>>::type, level_index<6, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, 2, 2>>::type, level_index<7, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, -2, 2>>::type, level_index<8, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, -1, 2>>::type, level_index<9, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, 1, 2>>::type, level_index<10, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, 2, 2>>::type, level_index<11, 2>>();
}

TEST(test_level, index_to_level) {
    ASSERT_TYPE_EQ<level<0, -2, 2>, typename index_to_level<level_index<0, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, -1, 2>, typename index_to_level<level_index<1, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, 1, 2>, typename index_to_level<level_index<2, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, 2, 2>, typename index_to_level<level_index<3, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, -2, 2>, typename index_to_level<level_index<4, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, -1, 2>, typename index_to_level<level_index<5, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, 1, 2>, typename index_to_level<level_index<6, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, 2, 2>, typename index_to_level<level_index<7, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, -2, 2>, typename index_to_level<level_index<8, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, -1, 2>, typename index_to_level<level_index<9, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, 1, 2>, typename index_to_level<level_index<10, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, 2, 2>, typename index_to_level<level_index<11, 2>>::type>();
}
