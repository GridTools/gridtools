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
            auto shared_cuda_ptr = alloc.allocate(sizeof(float_type));

            float_type *ptr = static_cast<float_type *>(shared_cuda_ptr.get());
            float_type data;

            test_allocated<<<1, 1>>>(ptr);
            cudaMemcpy(&data, ptr, sizeof(float_type), cudaMemcpyDeviceToHost);
            ASSERT_EQ(1, data);
        }
    } // namespace
} // namespace gridtools
