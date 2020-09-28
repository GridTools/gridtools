/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <set>

#include <gridtools/common/hugepage_alloc.hpp>

namespace gridtools {
    namespace {

        TEST(hugepage_alloc, alloc_free) {
            std::size_t n = 100;

            int *ptr = static_cast<int *>(hugepage_alloc(n * sizeof(int)));
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0);

            for (std::size_t i = 0; i < n; ++i) {
                ptr[i] = 0;
                EXPECT_EQ(ptr[i], 0);
            }

            hugepage_free(ptr);
        }

        TEST(hugepage_alloc, offsets) {
            // test shifting of the allocated data: hugepage_alloc guarantees that consecutive allocations return
            // pointers with different last 12bits to reduce number of cache set conflict misses
            std::set<std::uintptr_t> offsets;
            std::size_t checks = 7;
            for (std::size_t i = 0; i < checks; ++i) {
                double *ptr = static_cast<double *>(hugepage_alloc(sizeof(double)));
                offsets.insert(reinterpret_cast<std::uintptr_t>(ptr) & 0xfff);
                hugepage_free(ptr);
            }
#ifndef __cray__ // Cray clang version 10.0.2 seems to do a wrong optimization
            EXPECT_EQ(offsets.size(), checks);
#endif
        }

    } // namespace
} // namespace gridtools
