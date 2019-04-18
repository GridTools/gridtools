/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "test_expressions.cpp"

#include "../../cuda_test_helper.hpp"

TEST(test_expressions_cuda, add_accessors) {
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_add_accessors)), 3.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_sub_accessors)), -1.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_negate_accessors)), -1.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_plus_sign_accessors)), 1.f);
    EXPECT_FLOAT_EQ(
        gridtools::on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&test_with_parenthesis_accessors)), -3.f);
}
