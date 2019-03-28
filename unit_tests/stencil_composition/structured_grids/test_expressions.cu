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
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(MAKE_CONSTANT(test_add_accessors)), 3.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(MAKE_CONSTANT(test_sub_accessors)), -1.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(MAKE_CONSTANT(test_negate_accessors)), -1.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(MAKE_CONSTANT(test_plus_sign_accessors)), 1.f);
    EXPECT_FLOAT_EQ(gridtools::on_device::exec(MAKE_CONSTANT(test_with_parenthesis_accessors)), -3.f);
}
