/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

// The purpose of this set of tests is to guarantee that the offsets methods of the different specializations
// provided by the connectivity tables in from<>::to<>::with_color return a constexpr array
// It is not intended to check here the actual value of the offsets, this would only replicate the values coded
// in the tables

// From Cells to XXX
TEST(from_cell_to_cell, offsets_color0) {

    constexpr auto offsets = from< cells >::to< cells >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_cell, offsets_color1) {

    constexpr auto offsets = from< cells >::to< cells >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_edge, offsets_color0) {

    constexpr auto offsets = from< cells >::to< edges >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_edge, offsets_color1) {

    constexpr auto offsets = from< cells >::to< edges >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_cell_to_vertex, offsets_color0) {

    constexpr auto offsets = from< cells >::to< vertexes >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}
TEST(from_cell_to_vertex, offsets_color1) {

    constexpr auto offsets = from< cells >::to< vertexes >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

// From Edges to XXX
TEST(from_edge_to_cell, offsets_color0) {

    constexpr auto offsets = from< edges >::to< cells >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_cell, offsets_color1) {

    constexpr auto offsets = from< edges >::to< cells >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_cell, offsets_color2) {

    constexpr auto offsets = from< edges >::to< cells >::with_color< static_uint< 2 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color0) {

    constexpr auto offsets = from< edges >::to< edges >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color1) {

    constexpr auto offsets = from< edges >::to< edges >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_edge, offsets_color2) {

    constexpr auto offsets = from< edges >::to< edges >::with_color< static_uint< 2 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color0) {

    constexpr auto offsets = from< edges >::to< vertexes >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color1) {

    constexpr auto offsets = from< edges >::to< vertexes >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color2) {

    constexpr auto offsets = from< edges >::to< vertexes >::with_color< static_uint< 2 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

// From Vertexes to XXX
TEST(from_vertex_to_cell, offsets_color0) {

    constexpr auto offsets = from< vertexes >::to< cells >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_edge, offsets_color0) {

    constexpr auto offsets = from< vertexes >::to< edges >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_vertex, offsets_color0) {

    constexpr auto offsets = from< vertexes >::to< vertexes >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}
