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
#include <gridtools/stencil-composition/axis.hpp>

using namespace gridtools;

constexpr int level_offset_limit = 2;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

TEST(test_axis, ctor) {
    using axis_t = axis<2, 0, level_offset_limit>;
    auto axis_ = axis_t((uint_t)5, (uint_t)4);

    ASSERT_EQ(5, axis_.interval_size(0));
    ASSERT_EQ(4, axis_.interval_size(1));
}

TEST(test_axis, intervals) {
    using axis_t = axis<3, 0, level_offset_limit>;

    // full interval
    ASSERT_TYPE_EQ<interval<level_t<0, 1>, level_t<3, -1>>, axis_t::full_interval>();

    // intervals by id
    ASSERT_TYPE_EQ<interval<level_t<0, 1>, level_t<1, -1>>, axis_t::get_interval<0>>();
    ASSERT_TYPE_EQ<interval<level_t<1, 1>, level_t<2, -1>>, axis_t::get_interval<1>>();

    // hull of multiple intervals
    ASSERT_TYPE_EQ<interval<level_t<1, 1>, level_t<3, -1>>, axis_t::get_interval<1, 2>>();
}
