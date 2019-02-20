/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "cuda_test_helper.hpp"
#include <gridtools/common/host_device.hpp>
#include <gtest/gtest.h>

struct cuda_test_example1 {
    static bool GT_FUNCTION apply() {
        if (3 == 3)
            return true;
        else
            return false;
    }
};
struct cuda_test_example2 {
    static bool GT_FUNCTION apply(int a, int b) {
        if (a == b)
            return true;
        else
            return false;
    }
};

TEST(cuda_test, example1) { ASSERT_TRUE(cuda_test<cuda_test_example1>()); }
TEST(cuda_test, example2) { ASSERT_TRUE(cuda_test<cuda_test_example2>(3, 3)); }
