/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "test_gt_math.cpp"

#include "../cuda_test_helper.hpp"

namespace {

    TEST(math_cuda, test_fabs) { EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_fabs))); }
    TEST(math_cuda, test_abs) { EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_fabs))); }

    TEST(math_cuda, test_log) {
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_log<double>), 2.3, std::log(2.3)));
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_log<float>), 2.3f, std::log(2.3f)));
    }

    TEST(math_cuda, test_exp) {
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_exp<double>), 2.3, std::exp(2.3)));
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_exp<float>), 2.3f, std::exp(2.3f)));
    }

    TEST(math_cuda, test_pow) {
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_pow<double>), 2.3, std::pow(2.3, 2.3)));
        EXPECT_TRUE(on_device::exec(MAKE_CONSTANT(test_pow<float>), 2.3f, std::pow(2.3f, 2.3f)));
    }

} // namespace
