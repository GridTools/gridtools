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

    constexpr auto offsets = from< cells >::to< vertices >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}
TEST(from_cell_to_vertex, offsets_color1) {

    constexpr auto offsets = from< cells >::to< vertices >::with_color< static_uint< 1 > >::offsets();
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

    constexpr auto offsets = from< edges >::to< vertices >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color1) {

    constexpr auto offsets = from< edges >::to< vertices >::with_color< static_uint< 1 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_edge_to_vertex, offsets_color2) {

    constexpr auto offsets = from< edges >::to< vertices >::with_color< static_uint< 2 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

// From Vertexes to XXX
TEST(from_vertex_to_cell, offsets_color0) {

    constexpr auto offsets = from< vertices >::to< cells >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_edge, offsets_color0) {

    constexpr auto offsets = from< vertices >::to< edges >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}

TEST(from_vertex_to_vertex, offsets_color0) {

    constexpr auto offsets = from< vertices >::to< vertices >::with_color< static_uint< 0 > >::offsets();
    typedef static_int< offsets[0][0] > test_type;
    ASSERT_TRUE(test_type::value < 10);
}
