/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../cuda_test_helper.hpp"
#include <gridtools/stencil_composition/backend_cuda/shared_allocator.hpp>

#include <gtest/gtest.h>

namespace {
    template <typename PtrHolder>
    __device__ ptrdiff_t get_origin_offset(PtrHolder ptr_holder) {
        extern __shared__ char shm[];
        return reinterpret_cast<char *>(ptr_holder()) - shm;
    }

    TEST(shared_allocator, basic_test) {
        gridtools::shared_allocator allocator;
        EXPECT_EQ(0, allocator.size());

        using alloc1_t = char[14];
        auto alloc1 = allocator.allocate<alloc1_t>(7);
        EXPECT_GE(allocator.size(), 7 * sizeof(alloc1_t));

        using alloc2_t = double;
        auto old_size = allocator.size();
        auto alloc2 = allocator.allocate<alloc2_t>(4);
        EXPECT_GE(allocator.size(), old_size + 4 * sizeof(alloc2_t));

        using alloc3_t = double;
        old_size = allocator.size();
        auto alloc3 = allocator.allocate<alloc3_t>(1);
        EXPECT_GE(allocator.size(), old_size + sizeof(alloc3_t));

        auto offset1 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_origin_offset<decltype(alloc1)>), alloc1);
        auto offset2 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_origin_offset<decltype(alloc2)>), alloc2);
        auto offset3 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_origin_offset<decltype(alloc3)>), alloc3);

        // check alignment for all allocations
        EXPECT_EQ(offset1 % sizeof(alloc1_t), 0);
        EXPECT_EQ(offset2 % sizeof(alloc2_t), 0);
        EXPECT_EQ(offset3 % sizeof(alloc3_t), 0);

        // check that allocations are large enough
        EXPECT_GE(offset2 - offset1, 7 * sizeof(alloc1_t));
        EXPECT_GE(offset3 - offset2, 4 * sizeof(alloc2_t));
        EXPECT_GE(allocator.size() - offset3, 1 * sizeof(alloc3_t));
    }

    template <class T>
    ptrdiff_t get_offset(gridtools::shared_allocator const &allocator, T const &alloc1, T const &alloc2) {
        auto offset1 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_origin_offset<T>), alloc1);
        auto offset2 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_origin_offset<T>), alloc2);
        return offset2 - offset1;
    }

    TEST(shared_allocator, pointer_arithmetics) {
        gridtools::shared_allocator allocator;
        auto some_alloc = allocator.allocate<double>(32);
        auto another_alloc = allocator.allocate<double>(32);

        EXPECT_EQ(get_offset(allocator, another_alloc, another_alloc + 3), 3 * (int)sizeof(double));
    }

} // namespace
