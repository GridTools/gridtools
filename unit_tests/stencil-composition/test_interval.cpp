/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../test_helper.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil-composition/interval.hpp>

using namespace gridtools;

constexpr int level_offset_limit = 3;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

TEST(test_interval, modify) {
    using my_interval = interval<level_t<0, -1>, level_t<1, -1>>;

    ASSERT_TYPE_EQ<interval<level_t<0, -2>, level_t<1, -1>>, my_interval::modify<-1, 0>>();
    ASSERT_TYPE_EQ<interval<level_t<0, 1>, level_t<1, 1>>, my_interval::modify<1, 1>>();
    ASSERT_TYPE_EQ<interval<level_t<0, -3>, level_t<1, -1>>, my_interval::modify<-2, 0>>();
    ASSERT_TYPE_EQ<interval<level_t<0, 2>, level_t<1, 2>>, my_interval::modify<2, 2>>();
}
