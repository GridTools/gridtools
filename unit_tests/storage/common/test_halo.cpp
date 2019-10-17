/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/storage/common/halo.hpp>

#include "gtest/gtest.h"

#include <type_traits>

using namespace gridtools;

// test zero halo getter
static_assert(std::is_same<zero_halo<4>, halo<0, 0, 0, 0>>::value, "");
static_assert(std::is_same<zero_halo<3>, halo<0, 0, 0>>::value, "");
static_assert(std::is_same<zero_halo<2>, halo<0, 0>>::value, "");
static_assert(std::is_same<zero_halo<1>, halo<0>>::value, "");

// test value correctness
static_assert(halo<2, 3, 4>::at<0>() == 2, "");
static_assert(halo<2, 3, 4>::at<1>() == 3, "");
static_assert(halo<2, 3, 4>::at<2>() == 4, "");

TEST(dummy, dummy) {}
