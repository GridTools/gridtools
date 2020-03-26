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

#include <iostream>
#include <utility>

#include <gridtools/common/timer/timer.hpp>

#include "regression_fixture_impl.hpp"
#include "timer_select.hpp"

namespace gridtools {
    struct regression_fixture {
        template <class Comp>
        static void benchmark(Comp &&comp) {
            size_t steps = regression_domain::steps();
            if (steps == 0)
                return;
            comp();
            timer<timer_impl_t> timer = {"NoName"};
            for (size_t i = 0; i != steps; ++i) {
                regression_domain::flush_cache();
                timer.start();
                comp();
                timer.pause();
            }
            std::cout << timer.to_string() << std::endl;
        }
    };
} // namespace gridtools
