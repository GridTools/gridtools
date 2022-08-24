/*
 * GridTools
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/fn/sid_neighbor_table.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <gridtools/fn/unstructured.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {

        using sid_neighbor_table::as_neighbor_table;

        using edge_dim_t = integral_constant<int_t, 0>;
        using edge_to_cell_dim_t = integral_constant<int_t, 1>;

        TEST(sid_neighbor_table, correctness) {
            constexpr std::size_t numElements = 3;
            constexpr std::size_t numNeighbors = 2;
            std::int32_t contents[numElements][numNeighbors] = {{0, 1}, {10, 11}, {20, 21}};
            const auto table = as_neighbor_table<edge_dim_t, edge_to_cell_dim_t, numNeighbors>(contents);

            auto [n00, n01] = neighbor_table_neighbors(table, 0);
            auto [n10, n11] = neighbor_table_neighbors(table, 1);
            auto [n20, n21] = neighbor_table_neighbors(table, 2);
            EXPECT_EQ(n00, 0);
            EXPECT_EQ(n01, 1);
            EXPECT_EQ(n10, 10);
            EXPECT_EQ(n11, 11);
            EXPECT_EQ(n20, 20);
            EXPECT_EQ(n21, 21);
        }

    } // namespace
} // namespace gridtools::fn