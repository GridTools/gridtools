/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/fn/scan.hpp>

namespace gridtools::fn {
    namespace {
        struct sum_scan : fwd {
            static GT_FUNCTION consteval auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc /*+ deref(iter)*/; });
            }
        };

        struct sum_fold : fwd {
            static GT_FUNCTION consteval auto body() {
                return [](auto acc, auto const &iter) { return acc /*+ deref(iter)*/; };
            }
        };
    } // namespace
} // namespace gridtools::fn
