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
        gridtools::shared_allocator allocator;
        EXPECT_EQ(0, allocator.size());

        auto offset = allocator.allocate<14>(3 * 14);
        // check that there is no overlap with the previous allocation
        EXPECT_LE(0, offset);
        // check that there is enough space to fit the allocation
        EXPECT_GE(allocator.size(), offset + 3 * 14);
        // check alignment
        EXPECT_EQ(0, offset % 14);

        auto old_size = allocator.size();
        offset = allocator.allocate<8>(16 * 8);
        EXPECT_LE(old_size, offset);
        EXPECT_GE(allocator.size(), old_size + 16 * 8);
        EXPECT_EQ(0, offset % 8);

        old_size = allocator.size();
        offset = allocator.allocate<8>(1 * 8);
        EXPECT_LE(old_size, offset);
        EXPECT_GE(allocator.size(), old_size + 1 * 8);
        EXPECT_EQ(0, offset % 8);
    }
} // namespace
