/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <iostream>
#include <utility>

#include "../common/defs.hpp"
#include "../stencil-composition/axis.hpp"
#include "computation_fixture.hpp"
#include "regression_fixture_impl.hpp"

namespace gridtools {
    template <size_t HaloSize = 0, class Axis = axis<1>>
    class regression_fixture : public computation_fixture<HaloSize, Axis>, _impl::regression_fixture_base {
      public:
        regression_fixture() : computation_fixture<HaloSize, Axis>(s_d1, s_d2, s_d3) {}

        template <class... Args>
        void verify(Args &&... args) const {
            if (s_needs_verification)
                computation_fixture<HaloSize, Axis>::verify(std::forward<Args>(args)...);
        }

        template <class Comp>
        void benchmark(Comp &&comp) const {
            if (s_steps == 0)
                return;
            // we run a first time the stencil, since if there is data allocation before by other codes, the first run
            // of the stencil is very slow (we dont know why). The flusher should make sure we flush the cache
            comp.run();
            comp.reset_meter();
            for (size_t i = 0; i != s_steps; ++i) {
#ifndef __CUDACC__
                flush_cache();
#endif
                comp.run();
            }
            std::cout << comp.print_meter() << std::endl;
        }
    };
} // namespace gridtools
