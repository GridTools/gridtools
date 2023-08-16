/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
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
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using sid_neighbor_table::as_neighbor_table;

        using edge_dim_t = integral_constant<int_t, 0>;
        using edge_to_cell_dim_t = integral_constant<int_t, 1>;

        template <class Table>
        __device__ auto neighbor_table_neighbors_device(Table const &table, int index) -> array<int, 2> {
            return neighbor_table::neighbors(table, index);
        }

        TEST(sid_neighbor_table, correctness_cuda) {
            constexpr std::size_t num_elements = 3;
            constexpr std::size_t num_neighbors = 2;

            const int data[num_elements][num_neighbors] = {{0, 1}, {10, 11}, {20, 21}};
            const auto device_data = cuda_util::cuda_malloc<int>(num_elements * num_neighbors);
            GT_CUDA_CHECK(cudaMemcpy(device_data.get(), &data, sizeof data, cudaMemcpyHostToDevice));
            using dim_hymap_t = hymap::keys<edge_dim_t, edge_to_cell_dim_t>;
            auto contents = sid::synthetic()
                                .set<sid::property::origin>(sid::host_device::simple_ptr_holder(device_data.get()))
                                .set<sid::property::strides>(dim_hymap_t::make_values(num_neighbors, 1));

            const auto table = as_neighbor_table<edge_dim_t, edge_to_cell_dim_t, num_neighbors>(contents);
            using table_t = std::decay_t<decltype(table)>;

            auto [n00, n01] = on_device::exec(
                GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&neighbor_table_neighbors_device<table_t>), table, 0);
            auto [n10, n11] = on_device::exec(
                GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&neighbor_table_neighbors_device<table_t>), table, 1);
            auto [n20, n21] = on_device::exec(
                GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&neighbor_table_neighbors_device<table_t>), table, 2);
            EXPECT_EQ(n00, 0);
            EXPECT_EQ(n01, 1);
            EXPECT_EQ(n10, 10);
            EXPECT_EQ(n11, 11);
            EXPECT_EQ(n20, 20);
            EXPECT_EQ(n21, 21);
        }
    } // namespace
} // namespace gridtools::fn
