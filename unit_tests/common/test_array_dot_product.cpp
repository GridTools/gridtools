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
#include <gridtools/common/array_dot_product.hpp>
#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(array_dot_product, test_dot_product) {
    constexpr array<uint_t, 4> a{1, 2, 3, 4};
    constexpr array<uint_t, 4> b{1, 2, 3, 4};

    static_assert(array_dot_product(a, b) == 1 + 2 * 2 + 3 * 3 + 4 * 4, " ");
    ASSERT_EQ(array_dot_product(a, b), 1 + 2 * 2 + 3 * 3 + 4 * 4);
}
