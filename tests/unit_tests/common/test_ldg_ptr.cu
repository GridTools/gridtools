/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/ldg_ptr.hpp>

#include <gridtools/common/cuda_util.hpp>

#include <gtest/gtest.h>

#include <cuda_test_helper.hpp>

namespace gridtools {
    namespace {
        __device__ bool test_non_const_device() {
            float data[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

            auto ptr = as_ldg_ptr(&data[2]);
            if (*ptr != 2.0f)
                return false;
            if (ptr + 2 != as_ldg_ptr(&data[4]))
                return false;
            if (ptr - 2 != as_ldg_ptr(&data[0]))
                return false;
            if (*(ptr + 2) != 4.0f)
                return false;
            if (*(ptr - 2) != 0.0f)
                return false;
            if (*(++ptr) != 3.0f)
                return false;
            if (*(ptr++) != 3.0f)
                return false;
            if (*(ptr--) != 4.0f)
                return false;
            if (*(--ptr) != 2.0f)
                return false;
            *ptr = 5.0f;
            if (*ptr != 5.0f)
                return false;
            return true;
        }

        TEST(as_ldg_ptr, non_const_device) {
            EXPECT_TRUE(on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_non_const_device)));
        }

        __device__ bool test_const_device() {
            float const data[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

            auto ptr = as_ldg_ptr(&data[2]);
            if (*ptr != 2.0f)
                return false;
            if (ptr + 2 != as_ldg_ptr(&data[4]))
                return false;
            if (ptr - 2 != as_ldg_ptr(&data[0]))
                return false;
            if (*(ptr + 2) != 4.0f)
                return false;
            if (*(ptr - 2) != 0.0f)
                return false;
            if (*(++ptr) != 3.0f)
                return false;
            if (*(ptr++) != 3.0f)
                return false;
            if (*(ptr--) != 4.0f)
                return false;
            if (*(--ptr) != 2.0f)
                return false;
            return true;
        }

        TEST(as_ldg_ptr, const_device) {
            EXPECT_TRUE(on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_const_device)));
        }
    } // namespace
} // namespace gridtools

#include "test_ldg_ptr.cpp"
