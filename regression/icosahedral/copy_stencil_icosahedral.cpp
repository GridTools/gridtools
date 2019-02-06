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

template <uint_t>
struct functor_copy {
    using out = inout_accessor<0, enumtype::cells>;
    using in = in_accessor<1, enumtype::cells>;
    using arg_list = make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(in{});
    }
};

using copy_stencil_icosahedral = regression_fixture<>;

TEST_F(copy_stencil_icosahedral, test) {
    arg<0, cells> p_out;
    arg<1, cells> p_in;
    auto in = make_storage<cells>([](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; });
    auto out = make_storage<cells>();
    make_computation(p_out = out,
        p_in = in,
        make_multistage(
            enumtype::execute<enumtype::parallel>(), make_stage<functor_copy, topology_t, cells>(p_out, p_in)))
        .run();
    verify(in, out);
}
