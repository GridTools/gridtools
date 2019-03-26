/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/shared_allocator.hpp>

#include <gtest/gtest.h>

namespace {
    TEST(shared_allocator, test) {
        gridtools::SharedAllocator allocator;
        EXPECT_EQ(0, allocator.size());

        EXPECT_EQ(0, (allocator.template allocate<std::array<char, 14>>(3)));
        EXPECT_EQ(3 * 14, allocator.size());

        EXPECT_EQ(6, (allocator.template allocate<std::array<char, 8>>(16)));
        EXPECT_EQ(176, allocator.size());

        EXPECT_EQ(22, (allocator.template allocate<std::array<char, 8>>(1)));
        EXPECT_EQ(184, allocator.size());
    }
} // namespace
