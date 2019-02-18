/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "test_grid.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil-composition/axis.hpp>

constexpr int level_offset_limit = 3;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

TEST(test_grid, k_total_length) {
    static const int_t offset_from = -2;
    static const int_t offset_to = 2;

    uint_t splitter_begin = 5;
    uint_t splitter_end = 50;

    typedef interval<level_t<0, offset_from>, level_t<1, offset_to + 1>> axis;
    grid<axis> grid_(halo_descriptor{}, halo_descriptor{}, {splitter_begin, splitter_end});

    uint_t expected_total_length = (int_t)splitter_end - (int_t)splitter_begin - offset_from + offset_to;

    ASSERT_EQ(expected_total_length, grid_.k_total_length());
}

class test_grid_copy_ctor : public ::testing::Test {
  private:
    halo_descriptor halo_i;
    halo_descriptor halo_j;
    const uint_t splitter_0;
    const uint_t splitter_1;

  public:
    typedef interval<level_t<0, -1>, level_t<1, -1>> axis;
    grid<axis> grid_;

    test_grid_copy_ctor()
        : halo_i(1, 1, 1, 3, 5), halo_j(2, 2, 2, 7, 10), splitter_0(2), splitter_1(5),
          grid_(halo_i, halo_j, {splitter_0, splitter_1}) {}
};

TEST_F(test_grid_copy_ctor, copy_on_host) {
    grid<axis> copy(grid_);

    ASSERT_TRUE(test_grid_eq(grid_, copy));
}

TEST(test_grid, make_grid_makes_splitters_and_values) {
    halo_descriptor empty_{0, 0, 0, 0, 1};

    const uint_t interval1_size = 5;
    const uint_t interval2_size = 10;

    auto grid_ = make_grid(empty_, empty_, axis<2>((uint_t)5, (uint_t)10));

    ASSERT_EQ(3, grid_.value_list.size());

    ASSERT_EQ(0, grid_.value_list[0]);
    ASSERT_EQ(interval1_size, grid_.value_list[1]);
    ASSERT_EQ(interval1_size + interval2_size, grid_.value_list[2]);
}
