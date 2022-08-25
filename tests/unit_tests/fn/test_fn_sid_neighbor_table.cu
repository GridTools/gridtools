/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/fn/sid_neighbor_table.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include <cuda_test_helper.hpp>

namespace gridtools::fn {
    namespace {
        using sid_neighbor_table::as_neighbor_table;

        using edge_dim_t = integral_constant<int_t, 0>;
        using edge_to_cell_dim_t = integral_constant<int_t, 1>;

        template <class Table>
        __device__ auto neighbor_table_neighbors_device(const Table &table, size_t index) -> array<std::int32_t, 2> {
            return neighbor_table_neighbors(table, index);
        }
        constexpr std::size_t num_elements = 3;
        constexpr std::size_t num_neighbors = 2;
        __device__ std::int32_t contents[num_elements][num_neighbors] = {{0, 1}, {10, 11}, {20, 21}};

        TEST(sid_neighbor_table, correctness_cuda) {
            const auto table = as_neighbor_table<edge_dim_t, edge_to_cell_dim_t, num_neighbors>(contents);

            const auto instantiation = &neighbor_table_neighbors_device<std::decay_t<decltype(table)>>;

            auto [n00, n01] = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(instantiation), table, 0);
            auto [n10, n11] = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(instantiation), table, 1);
            auto [n20, n21] = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(instantiation), table, 2);
            EXPECT_EQ(n00, 0);
            EXPECT_EQ(n01, 1);
            EXPECT_EQ(n10, 10);
            EXPECT_EQ(n11, 11);
            EXPECT_EQ(n20, 20);
            EXPECT_EQ(n21, 21);
        }
    } // namespace
} // namespace gridtools::fn