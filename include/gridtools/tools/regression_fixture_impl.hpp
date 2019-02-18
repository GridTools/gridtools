/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/defs.hpp"

namespace gridtools {
    namespace _impl {
        class regression_fixture_base {
          protected:
            static uint_t s_d1;
            static uint_t s_d2;
            static uint_t s_d3;
            static uint_t s_steps;
            static bool s_needs_verification;

            static void flush_cache();

          public:
            static void init(int argc, char **argv);
        };
    } // namespace _impl
} // namespace gridtools
