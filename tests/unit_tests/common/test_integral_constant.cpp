/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/integral_constant.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        static_assert(0_c == 0, "");
        static_assert(42_c == 42, "");
        static_assert(-12345_c == -12345, "");

        static_assert(0b100_c == 0b100, "");
        static_assert(0100_c == 0100, "");
        static_assert(0xDEAD_c == 0xDEAD, "");

        static_assert(1'000_c == 1000, "");

        static_assert(2_c + 3_c == 5_c, "");
    } // namespace
} // namespace gridtools
