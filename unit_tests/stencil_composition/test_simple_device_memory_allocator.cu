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
        __global__ void test_allocated(PtrHolder testee, bool *result) {
            auto &ref = *testee();
            ref = 1.;
            *result = ref == 1.;
        }

        TEST(simple_device_memory_allocator, test) {
            simple_device_memory_allocator alloc;
            auto ptr_holder = alloc.allocate<float_type>(1);

            bool result = false;
            bool *dev_result;
            GT_CUDA_CHECK(cudaMalloc(&dev_result, sizeof(bool)));
            GT_CUDA_CHECK(cudaMemcpy(dev_result, &result, sizeof(bool), cudaMemcpyHostToDevice));

            test_allocated<<<1, 1>>>(ptr_holder, dev_result);
            GT_CUDA_CHECK(cudaDeviceSynchronize());

            GT_CUDA_CHECK(cudaMemcpy(&result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost));
            ASSERT_TRUE(result);
        }
    } // namespace
} // namespace gridtools
