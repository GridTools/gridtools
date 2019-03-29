/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/cuda_allocator.hpp>
#include <gridtools/tools/backend_select.hpp>

#include "../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        __global__ void test_allocated(float_type *data) { *data = 1; }

        TEST(simple_cuda_allocator, test) {
            // TODO use test functionality
            cuda_allocator alloc;
            auto ptr_holder = alloc.allocate<float_type>(1);

            float_type *ptr = ptr_holder();
            float_type data;

            test_allocated<<<1, 1>>>(ptr);
            cudaMemcpy(&data, ptr, sizeof(float_type), cudaMemcpyDeviceToHost);
            ASSERT_EQ(1, data);
        }
    } // namespace
} // namespace gridtools
