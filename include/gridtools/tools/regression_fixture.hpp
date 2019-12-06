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

#include "../common/timer/timer.hpp"
#include "../stencil_composition/axis.hpp"
#include "backend_select.hpp"
#include "grid_fixture.hpp"
#include "regression_fixture_impl.hpp"

namespace gridtools {
    template <class Fixture>
    struct regression_fixture_templ : Fixture, private _impl::regression_fixture_base {
        regression_fixture_templ() : Fixture(s_d1, s_d2, s_d3) {}

        template <class... Args>
        void verify(Args &&... args) const {
            if (s_needs_verification)
                Fixture::verify(std::forward<Args>(args)...);
        }

        template <class Comp>
        void benchmark(Comp &&comp) const {
            if (s_steps == 0)
                return;
            // we run a first time the stencil, since if there is data allocation before by other codes, the first run
            // of the stencil is very slow (we dont know why). The flusher should make sure we flush the cache
            comp();
            timer<timer_impl_t> timer = {"NoName"};
            for (size_t i = 0; i != s_steps; ++i) {
#ifndef __CUDACC__
                flush_cache();
#endif
                timer.start();
                comp();
                timer.pause();
            }
            std::cout << timer.to_string() << std::endl;
        }
    };
} // namespace gridtools
