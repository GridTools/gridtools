/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"

#include <gridtools/stencil_composition/stencil_composition.hpp>

using namespace gridtools;
using namespace enumtype;

// The purpose of this set of tests is to guarantee that the offsets methods of the different specializations
// provided by the connectivity tables in from<>::to<>::with_color return a constexpr array
// It is not intended to check here the actual value of the offsets, this would only replicate the values coded
// in the tables

// From Cells to XXX
TEST(from_cell_to_cell, offsets_color0) {

    constexpr auto offsets = from<cells>::to<cells>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_cell, offsets_color1) {

    constexpr auto offsets = from<cells>::to<cells>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_edge, offsets_color0) {

    constexpr auto offsets = from<cells>::to<edges>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_edge, offsets_color1) {

    constexpr auto offsets = from<cells>::to<edges>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_vertex, offsets_color0) {

    constexpr auto offsets = from<cells>::to<vertices>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}
TEST(from_cell_to_vertex, offsets_color1) {

    constexpr auto offsets = from<cells>::to<vertices>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

// From Edges to XXX
TEST(from_edge_to_cell, offsets_color0) {

    constexpr auto offsets = from<edges>::to<cells>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_cell, offsets_color1) {

    constexpr auto offsets = from<edges>::to<cells>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_cell, offsets_color2) {

    constexpr auto offsets = from<edges>::to<cells>::with_color<static_uint<2>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color0) {

    constexpr auto offsets = from<edges>::to<edges>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color1) {

    constexpr auto offsets = from<edges>::to<edges>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color2) {

    constexpr auto offsets = from<edges>::to<edges>::with_color<static_uint<2>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color0) {

    constexpr auto offsets = from<edges>::to<vertices>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color1) {

    constexpr auto offsets = from<edges>::to<vertices>::with_color<static_uint<1>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color2) {

    constexpr auto offsets = from<edges>::to<vertices>::with_color<static_uint<2>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

// From Vertexes to XXX
TEST(from_vertex_to_cell, offsets_color0) {

    constexpr auto offsets = from<vertices>::to<cells>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_edge, offsets_color0) {

    constexpr auto offsets = from<vertices>::to<edges>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_vertex, offsets_color0) {

    constexpr auto offsets = from<vertices>::to<vertices>::with_color<static_uint<0>>::offsets();
    typedef static_int<offsets[0][0]> test_type;
    ASSERT_TRUE(test_type::value < 10);
}
