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

using icosahedral_topology_t =
    gridtools::icosahedral_topology< gridtools::backend< enumtype::Host, enumtype::icosahedral, enumtype::Block > >;

class connectivity_index_test : public ::testing::TestWithParam< int > {
    // You can implement all the usual fixture class members here.
    // To access the test parameter, call GetParam() from class
    // TestWithParam<T>.
};

TEST_P(connectivity_index_test, cell_to_cell) {

    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (uint_t i = 0; i < d3; i += 2) {
        // cell # 50
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), cells(), gridtools::static_uint< 0 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
            array< uint_t, 3 >{42 * d3 + i, 58 * d3 + i, 57 * d3 + i})));
        // cell # 58
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), cells(), gridtools::static_uint< 1 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
            array< uint_t, 3 >{50 * d3 + i, 51 * d3 + i, 66 * d3 + i})));
        // cell # 35
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), cells(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 3 >{42 * d3 + i, 43 * d3 + i, 27 * d3 + i})));
        // cell # 45
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), cells(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)5, (uint_t)i}),
            array< uint_t, 3 >{53 * d3 + i, 38 * d3 + i, 37 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, cell_to_edge) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // cell # 50
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), edges(), gridtools::static_uint< 0 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
            array< uint_t, 3 >{74 * d3 + i, 82 * d3 + i, 90 * d3 + i})));
        // cell # 58
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), edges(), gridtools::static_uint< 1 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
            array< uint_t, 3 >{90 * d3 + i, 75 * d3 + i, 106 * d3 + i})));
        // cell # 35
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), edges(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 3 >{51 * d3 + i, 59 * d3 + i, 67 * d3 + i})));
        // cell # 45
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(cells(), edges(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)5, (uint_t)i}),
            array< uint_t, 3 >{69 * d3 + i, 54 * d3 + i, 85 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, cell_to_vertex) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // cell # 50
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               cells(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
                array< uint_t, 3 >{26 * d3 + i, 27 * d3 + i, 34 * d3 + i})));
        // cell # 58
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               cells(), vertices(), gridtools::static_uint< 1 >(), {(uint_t)3, (uint_t)2, (uint_t)i}),
                array< uint_t, 3 >{34 * d3 + i, 27 * d3 + i, 35 * d3 + i})));
        // cell # 35
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               cells(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 3 >{19 * d3 + i, 20 * d3 + i, 27 * d3 + i})));
        // cell # 45
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               cells(), vertices(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)5, (uint_t)i}),
                array< uint_t, 3 >{22 * d3 + i, 29 * d3 + i, 30 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, edge_to_edge) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // edge # 51
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{66 * d3 + i, 59 * d3 + i, 67 * d3 + i, 82 * d3 + i})));
        // edge # 59
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{43 * d3 + i, 28 * d3 + i, 67 * d3 + i, 51 * d3 + i})));
        // edge # 67
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 2 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{51 * d3 + i, 59 * d3 + i, 52 * d3 + i, 83 * d3 + i})));

        // edge # 123
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{138 * d3 + i, 131 * d3 + i, 139 * d3 + i, 154 * d3 + i})));
        // edge # 131
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 1 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{115 * d3 + i, 100 * d3 + i, 139 * d3 + i, 123 * d3 + i})));
        // edge # 139
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), edges(), gridtools::static_uint< 2 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 4 >{123 * d3 + i, 131 * d3 + i, 124 * d3 + i, 155 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, edge_to_cell) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // edge # 51
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{42 * d3 + i, 35 * d3 + i})));
        // edge # 59
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{27 * d3 + i, 35 * d3 + i})));
        // edge # 67
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 2 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{35 * d3 + i, 43 * d3 + i})));

        // edge # 123
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{90 * d3 + i, 83 * d3 + i})));
        // edge # 131
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 1 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{75 * d3 + i, 83 * d3 + i})));
        // edge # 139
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(edges(), cells(), gridtools::static_uint< 2 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 2 >{83 * d3 + i, 91 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, edge_to_vertex) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // edge # 51
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{19 * d3 + i, 27 * d3 + i})));

        // edge # 59
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 1 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{19 * d3 + i, 20 * d3 + i})));
        // edge # 67
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 2 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{20 * d3 + i, 27 * d3 + i})));

        // edge # 123
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{43 * d3 + i, 51 * d3 + i})));
        // edge # 131
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 1 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{43 * d3 + i, 44 * d3 + i})));
        // edge # 139
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               edges(), vertices(), gridtools::static_uint< 2 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
                array< uint_t, 2 >{44 * d3 + i, 51 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, vertex_to_vertex) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // vertex # 21
        ASSERT_TRUE((
            same_elements(grid.connectivity_index(
                              vertices(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 6 >{18 * d3 + i, 11 * d3 + i, 12 * d3 + i, 20 * d3 + i, 27 * d3 + i, 26 * d3 + i})));
        // vertex # 48
        ASSERT_TRUE((
            same_elements(grid.connectivity_index(
                              vertices(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
                array< uint_t, 6 >{35 * d3 + i, 36 * d3 + i, 44 * d3 + i, 51 * d3 + i, 50 * d3 + i, 42 * d3 + i})));
        // vertex # 60
        ASSERT_TRUE((
            same_elements(grid.connectivity_index(
                              vertices(), vertices(), gridtools::static_uint< 0 >(), {(uint_t)6, (uint_t)6, (uint_t)i}),
                array< uint_t, 6 >{53 * d3 + i, 46 * d3 + i, 47 * d3 + i, 55 * d3 + i, 62 * d3 + i, 61 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, vertex_to_cells) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // vertex # 21
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               vertices(), cells(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 6 >{26 * d3 + i, 19 * d3 + i, 27 * d3 + i, 35 * d3 + i, 42 * d3 + i, 34 * d3 + i})));
        // vertex # 48
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               vertices(), cells(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
                array< uint_t, 6 >{74 * d3 + i, 67 * d3 + i, 75 * d3 + i, 83 * d3 + i, 90 * d3 + i, 82 * d3 + i})));
        // vertex # 60
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               vertices(), cells(), gridtools::static_uint< 0 >(), {(uint_t)6, (uint_t)6, (uint_t)i}),
                array< uint_t, 6 >{93 * d3 + i, 86 * d3 + i, 94 * d3 + i, 101 * d3 + i, 102 * d3 + i, 109 * d3 + i})));
    }
}

TEST_P(connectivity_index_test, vertex_to_edges) {
    const uint_t d3 = GetParam();
    icosahedral_topology_t grid(8, 8, d3);

    for (int i = 0; i < d3; i += 2) {
        // vertex # 21
        ASSERT_TRUE(
            (same_elements(grid.connectivity_index(
                               vertices(), edges(), gridtools::static_uint< 0 >(), {(uint_t)2, (uint_t)3, (uint_t)i}),
                array< uint_t, 6 >{58 * d3 + i, 27 * d3 + i, 43 * d3 + i, 59 * d3 + i, 51 * d3 + i, 66 * d3 + i})));
        // vertex # 48
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(
                vertices(), edges(), gridtools::static_uint< 0 >(), {(uint_t)5, (uint_t)3, (uint_t)i}),
            array< uint_t, 6 >{130 * d3 + i, 99 * d3 + i, 115 * d3 + i, 131 * d3 + i, 123 * d3 + i, 138 * d3 + i})));
        // vertex # 60
        ASSERT_TRUE((same_elements(
            grid.connectivity_index(
                vertices(), edges(), gridtools::static_uint< 0 >(), {(uint_t)6, (uint_t)6, (uint_t)i}),
            array< uint_t, 6 >{157 * d3 + i, 126 * d3 + i, 142 * d3 + i, 158 * d3 + i, 150 * d3 + i, 165 * d3 + i})));
    }
}

INSTANTIATE_TEST_CASE_P(InstantiationName, connectivity_index_test, ::testing::Values(2, 5, 9));
