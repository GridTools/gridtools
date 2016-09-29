/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "stencil-composition/interval.hpp"

using namespace gridtools;

TEST(test_interval, join_interval_in_order) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, 1 >;
    using level3 = level< 1, -1 >;
    using level4 = level< 2, 1 >;

    using interval1 = interval< level1, level2 >;
    using interval2 = interval< level3, level4 >;

    using joined_interval = join_interval< interval1, interval2 >::type;

    ::testing::StaticAssertTypeEq< joined_interval::FromLevel, interval1::FromLevel >();
    ::testing::StaticAssertTypeEq< joined_interval::ToLevel, interval2::ToLevel >();
}

TEST(test_interval, join_interval_t) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, 1 >;
    using level3 = level< 1, -1 >;
    using level4 = level< 2, 1 >;

    using interval1 = interval< level1, level2 >;
    using interval2 = interval< level3, level4 >;

    using joined_interval = join_interval_t< interval1, interval2 >;

    ::testing::StaticAssertTypeEq< joined_interval::FromLevel, interval1::FromLevel >();
    ::testing::StaticAssertTypeEq< joined_interval::ToLevel, interval2::ToLevel >();
}

// TEST(test_interval, join_interval_non_contiguous) {
//    using level1 = level< 0, -1 >;
//    using level2 = level< 1, -1 >;
//    using level3 = level< 1, 2 >;
//    using level4 = level< 2, 1 >;
//
//    using interval1 = interval< level1, level2 >;
//    using interval2 = interval< level3, level4 >;
//
//    using joined_interval = join_interval< interval1, interval2 >::type;
//
//    ::testing::StaticAssertTypeEq< joined_interval::FromLevel, interval1::FromLevel >();
//    ::testing::StaticAssertTypeEq< joined_interval::ToLevel, interval2::ToLevel >();
//}

TEST(test_interval, is_contiguous_non_intersecting) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -1 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    ASSERT_TRUE((join_interval_is_contiguous< leftInterval, rightInterval >::value));
}

TEST(test_interval, is_contiguous_intersecting) {
    using level1 = level< 0, -1 >;
    using level2 = level< 2, -1 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    ASSERT_TRUE((join_interval_is_contiguous< leftInterval, rightInterval >::value));
}

TEST(test_interval, is_not_contiguous) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -2 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    ASSERT_FALSE((join_interval_is_contiguous< leftInterval, rightInterval >::value));
}

TEST(test_interval, is_subset_of) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -1 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using outerInterval = interval< level1, level4 >;
    using innerInterval = interval< level2, level3 >;

    ASSERT_FALSE(check_interval< outerInterval >::is_subset_of< innerInterval >::value);
    ASSERT_TRUE(check_interval< innerInterval >::is_subset_of< outerInterval >::value);
}

TEST(test_interval, is_subset_of_same_splitter) {
    using level1 = level< 0, -2 >;
    using level2 = level< 0, -1 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 1, 2 >;

    using outerInterval = interval< level1, level4 >;
    using innerInterval = interval< level2, level3 >;

    ASSERT_FALSE(check_interval< outerInterval >::is_subset_of< innerInterval >::value);
    ASSERT_TRUE(check_interval< innerInterval >::is_subset_of< outerInterval >::value);
}

TEST(test_interval, is_subset_of_with_equal_intervals) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -1 >;

    using interval1 = interval< level1, level2 >;
    using interval2 = interval1;

    ASSERT_TRUE(check_interval< interval1 >::is_subset_of< interval2 >::value);
    ASSERT_TRUE(check_interval< interval2 >::is_subset_of< interval1 >::value);
}

TEST(test_make_axis, two_intervals_in_order) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -2 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    using axis = make_axis< leftInterval, rightInterval >::type;

    ASSERT_TRUE(check_interval< leftInterval >::is_strict_subset_of< axis >::value);
    ASSERT_TRUE(check_interval< rightInterval >::is_strict_subset_of< axis >::value);
}

TEST(test_make_axis, make_axis_t) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -2 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    using axis = make_axis_t< leftInterval, rightInterval >;

    ASSERT_TRUE(check_interval< leftInterval >::is_strict_subset_of< axis >::value);
    ASSERT_TRUE(check_interval< rightInterval >::is_strict_subset_of< axis >::value);
}

TEST(test_make_axis, two_intervals_reverse_order) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -2 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;

    using leftInterval = interval< level1, level2 >;
    using rightInterval = interval< level3, level4 >;

    using axis = make_axis< rightInterval, leftInterval >::type;

    ASSERT_TRUE(check_interval< leftInterval >::is_strict_subset_of< axis >::value);
    ASSERT_TRUE(check_interval< rightInterval >::is_strict_subset_of< axis >::value);
}

TEST(test_make_axis, three_intervals) {
    using level1 = level< 0, -1 >;
    using level2 = level< 1, -2 >;
    using level3 = level< 1, 1 >;
    using level4 = level< 2, 1 >;
    using level5 = level< 3, -1 >;

    using interval1 = interval< level1, level2 >;
    using interval2 = interval< level2, level5 >;
    using interval3 = interval< level3, level4 >;

    using axis = make_axis< interval2, interval1, interval3 >::type;

    ASSERT_TRUE(check_interval< interval1 >::is_strict_subset_of< axis >::value);
    ASSERT_TRUE(check_interval< interval2 >::is_strict_subset_of< axis >::value);
    ASSERT_TRUE(check_interval< interval3 >::is_strict_subset_of< axis >::value);
}

// TEST(test_make_axis, interval_starting_at_index_0) {
//    using level1 = level< 1, -3 >;
//    using level2 = level< 1, -2 >;
//    using level3 = level< 1, 1 >;
//    using level4 = level< 2, 3 >;
//
//    using leftInterval = interval< level1, level2 >;
//    using rightInterval = interval< level3, level4 >;
//
//    using axis = make_axis< leftInterval, rightInterval >::type;
//
//    ASSERT_TRUE(check_interval< leftInterval >::is_strict_subset_of< axis >::value);
//    ASSERT_TRUE(check_interval< rightInterval >::is_strict_subset_of< axis >::value);
//}
