/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
