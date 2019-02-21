/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>

#include "gtest/gtest.h"
#include <gridtools/common/numerics.hpp>

using namespace gridtools;

TEST(numerics, pow3) {
    constexpr int x0 = _impl::static_pow3<0>::value;
    constexpr int x1 = _impl::static_pow3<1>::value;
    constexpr int x2 = _impl::static_pow3<2>::value;
    constexpr int x3 = _impl::static_pow3<3>::value;
    constexpr int x4 = _impl::static_pow3<4>::value;
    EXPECT_EQ(x0, 1);
    EXPECT_EQ(x1, 3);
    EXPECT_EQ(x2, 9);
    EXPECT_EQ(x3, 27);
    EXPECT_EQ(x4, 81);
}
