/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/allocator.hpp>

#include <cstddef>
#include <memory>

#include <gtest/gtest.h>

#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {

        TEST(allocator, allocates_once_per_request) {
            int count = 0;
            auto alloc = sid::allocator([&count](size_t size) {
                ++count;
                return std::make_unique<char[]>(size);
            });

            allocate(alloc, meta::lazy::id<double>{}, 3);
            EXPECT_EQ(1, count);

            allocate(alloc, meta::lazy::id<double>{}, 3);
            EXPECT_EQ(2, count);
        }
    } // namespace
} // namespace gridtools
