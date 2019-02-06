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
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct lap {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;
    using arg_list = make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = 4 * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

using laplacian = regression_fixture<1>;

TEST_F(laplacian, test) {
    auto in = [](int_t, int_t, int_t) { return -1.; };
    auto ref = [in](int_t i, int_t j, int_t k) {
        return 4 * in(i, j, k) - (in(i + 1, j, k) + in(i, j + 1, k) + in(i - 1, j, k) + in(i, j - 1, k));
    };
    auto out = make_storage(-7.3);

    make_computation(p_0 = out,
        p_1 = make_storage(in),
        make_multistage(enumtype::execute<enumtype::forward>(), make_stage<lap>(p_0, p_1)))
        .run();

    verify(make_storage(ref), out);
}
