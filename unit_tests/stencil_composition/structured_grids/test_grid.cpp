/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/grid.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/axis.hpp>

constexpr int level_offset_limit = 3;

using namespace gridtools;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

TEST(test_grid, k_total_length) {
    static constexpr int_t offset_from = -2;
    static constexpr int_t offset_to = 2;

    int_t interval_len = 45;

    using axis = interval<level_t<0, offset_from>, level_t<1, offset_to + 1>>;
    grid<axis> testee(halo_descriptor{}, halo_descriptor{}, {interval_len});

    EXPECT_EQ(interval_len - offset_from + offset_to, testee.k_total_length());
}

TEST(test_grid, make_grid_makes_splitters_and_values) {
    halo_descriptor empty_{0, 0, 0, 0, 1};

    const int_t interval1_size = 5;
    const int_t interval2_size = 10;

    auto testee = make_grid(empty_, empty_, axis<2>{interval1_size, interval2_size});

    EXPECT_EQ(0, (testee.value_at<level_t<0, 1>>()));
    EXPECT_EQ(interval1_size, (testee.value_at<level_t<1, 1>>()));
    EXPECT_EQ(interval1_size + interval2_size, (testee.value_at<level_t<2, 1>>()));
}
