/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <cstdlib>

#include <gridtools/common/defs.hpp>

namespace gridtools {
    class regression_domain {
        static std::array<uint_t, 3> s_d;
        static uint_t s_steps;
        static bool s_needs_verification;

      public:
        static auto d(size_t i) { return s_d[i]; }
        static auto steps() { return s_steps; }
        static auto needs_verification() { return s_needs_verification; }

        static void flush_cache();

        static bool init(int argc, char **argv);
    };
} // namespace gridtools
