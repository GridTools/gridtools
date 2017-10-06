/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include "gtest/gtest.h"
#include "test_grid.hpp"
#include "stencil-composition/axis.hpp"

TEST(test_grid, k_total_length) {
    static const int_t offset_from = -2;
    static const int_t offset_to = 2;

    uint_t splitter_begin = 5;
    uint_t splitter_end = 50;

    typedef interval< level< 0, offset_from >, level< 1, offset_to + 1 > > axis;
    grid< axis > grid_(halo_descriptor{}, halo_descriptor{});
    grid_.value_list[0] = splitter_begin;
    grid_.value_list[1] = splitter_end;

    uint_t expected_total_length = (int_t)splitter_end - (int_t)splitter_begin - offset_from + offset_to;

    ASSERT_EQ(expected_total_length, grid_.k_total_length());
}

class test_grid_copy_ctor : public ::testing::Test {
  private:
    halo_descriptor halo_i;
    halo_descriptor halo_j;
    const int splitter_0;
    const int splitter_1;

  public:
    typedef interval< level< 0, -1 >, level< 1, -1 > > axis;
    grid< axis > grid_;

    test_grid_copy_ctor()
        : halo_i(1, 1, 1, 3, 5), halo_j(2, 2, 2, 7, 10), splitter_0(2), splitter_1(5), grid_(halo_i, halo_j) {
        grid_.value_list[0] = splitter_0;
        grid_.value_list[1] = splitter_1;
    }
};

TEST_F(test_grid_copy_ctor, copy_on_host) {
    grid< axis > copy(grid_);

    ASSERT_TRUE(test_grid_eq(grid_, copy));
}

TEST(test_grid, make_grid_makes_splitters_and_values) {
    halo_descriptor empty_{0, 0, 0, 0, 1};

    const uint_t interval1_size = 5;
    const uint_t interval2_size = 10;

    auto grid_ = make_grid(empty_, empty_, axis< 2 >((uint_t)5, (uint_t)10));

    ASSERT_EQ(3, grid_.value_list.size());

    ASSERT_EQ(0, grid_.value_list[0]);
    ASSERT_EQ(interval1_size - 1, grid_.value_list[1]);
    ASSERT_EQ(interval1_size + interval2_size - 1, grid_.value_list[2]);
}

TEST(test_grid, grid_makes_splitters_and_values) {
    halo_descriptor empty_{0, 0, 0, 0, 1};

    const uint_t interval1_size = 5;
    const uint_t interval2_size = 10;

    grid< axis< 2 >::axis_interval_t > grid_(empty_, empty_, axis< 2 >((uint_t)5, (uint_t)10));

    ASSERT_EQ(3, grid_.value_list.size());

    ASSERT_EQ(0, grid_.value_list[0]);
    ASSERT_EQ(interval1_size - 1, grid_.value_list[1]);
    ASSERT_EQ(interval1_size + interval2_size - 1, grid_.value_list[2]);
}
