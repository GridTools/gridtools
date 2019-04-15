/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/tools/backend_select.hpp>

#include "../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        template <typename PtrHolder>
        __global__ void test_allocated(PtrHolder data) {
            *(data()) = 1;
        }

        TEST(simple_device_memory_allocator, test) {
            simple_device_memory_allocator alloc;
            auto ptr_holder = alloc.allocate<float_type>(1);

            test_allocated<<<1, 1>>>(ptr_holder);
            float_type data;
            cudaMemcpy(&data, alloc.ptrs()[0].get(), sizeof(float_type), cudaMemcpyDeviceToHost);
            ASSERT_EQ(1, data);
        }
    } // namespace
} // namespace gridtools
