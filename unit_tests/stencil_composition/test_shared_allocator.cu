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
    __device__ uint64_t get_ptr(PtrHolder ptr_holder) {
        return reinterpret_cast<uint64_t>(ptr_holder());
    }

    TEST(shared_allocator, alignment) {
        gridtools::shared_allocator allocator;
        EXPECT_EQ(0, allocator.size());

        using alloc1_t = char[14];
        auto alloc1 = allocator.allocate<alloc1_t>(7);

        using alloc2_t = double;
        auto alloc2 = allocator.allocate<alloc2_t>(4);

        using alloc3_t = double;
        auto alloc3 = allocator.allocate<alloc3_t>(1);

        auto ptr1 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_ptr<decltype(alloc1)>), alloc1);
        auto ptr2 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_ptr<decltype(alloc2)>), alloc2);
        auto ptr3 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_ptr<decltype(alloc3)>), alloc3);

        // check alignment for all allocations
        EXPECT_EQ(ptr1 % alignof(alloc1_t), 0);
        EXPECT_EQ(ptr2 % alignof(alloc2_t), 0);
        EXPECT_EQ(ptr3 % alignof(alloc3_t), 0);
    }

    template <class T>
    ptrdiff_t get_offset(gridtools::shared_allocator const &allocator, T const &alloc1, T const &alloc2) {
        auto offset1 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_ptr<T>), alloc1);
        auto offset2 = gridtools::on_device::exec_with_shared_memory(
            allocator.size(), MAKE_CONSTANT(get_ptr<T>), alloc2);
        return offset2 - offset1;
    }

    TEST(shared_allocator, pointer_arithmetics) {
        gridtools::shared_allocator allocator;
        auto some_alloc = allocator.allocate<double>(32);
        auto another_alloc = allocator.allocate<double>(32);

        EXPECT_EQ(get_offset(allocator, another_alloc, another_alloc + 3), 3 * (int)sizeof(double));
    }

    template <class PtrHolderFloat, class PtrHolderDouble>
    __device__ int fill_and_check_test(PtrHolderFloat alloc1, PtrHolderDouble alloc2, bool *result) {
        static_assert(std::is_same<decltype(alloc1()), float *>::value, "");
        static_assert(std::is_same<decltype(alloc2()), double *>::value, "");

        auto ptr1 = alloc1();
        auto ptr2 = alloc2();

        ptr1[threadIdx.x] = 100 * blockIdx.x + threadIdx.x;
        ptr2[threadIdx.x] = 10000 + 100 * blockIdx.x + threadIdx.x;
        __syncthreads();

        if (threadIdx.x == 0) {
            bool local_result = true;
            for (int i = 0; i < 32; ++i)
                local_result &= (ptr1[i] == 100 * blockIdx.x + i && ptr2[i] == 10000 + 100 * blockIdx.x + i);

            result[blockIdx.x] = local_result;
        }
        return 0;
    }

    TEST(shared_allocator, fill_and_check) {
        gridtools::shared_allocator allocator;
        auto float_allocation = allocator.allocate<float>(32);
        auto double_allocation = allocator.allocate<double>(32);

        bool *result;
        cudaMallocManaged(&result, 2 * sizeof(bool));

        gridtools::on_device::exec_with_shared_memory<2, 32>(allocator.size(),
            MAKE_CONSTANT((fill_and_check_test<decltype(float_allocation), decltype(double_allocation)>)),
            float_allocation,
            double_allocation,
            result);

        EXPECT_TRUE(result[0]);
        EXPECT_TRUE(result[1]);

        cudaFree(result);
    }

} // namespace
