/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"
#include <gridtools/common/error.hpp>

using namespace gridtools;

TEST(err_or_return, true_condition) {
    size_t return_val = 1;
    bool true_condition = true;

    ASSERT_EQ(return_val, error_or_return(true_condition, return_val, ""));
}

TEST(err_or_return, false_condition) {
    size_t return_val = 1;
    bool false_condition = false;

    ASSERT_ANY_THROW(error_or_return(false_condition, return_val, ""));
}

constexpr size_t call_error_or_return(bool condition, size_t return_val) {
    return error_or_return(condition, return_val, "");
}

TEST(err_or_return, in_constexpr_context) {
    const size_t return_val = 1;
    const bool true_condition = true;

    constexpr size_t result = call_error_or_return(true_condition, return_val);
    ASSERT_EQ(return_val, result);
}
