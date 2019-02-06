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

// TODO(fthaler): merge this with horizontal_diffusion_functions and benchmark all the horizontal_diffusion versions

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;

struct lap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;

    using arg_list = make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) =
            float_type{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 2, -1, 1>>;

    using arg_list = make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto lap_hi = call<lap_function>::with(eval, in(1, 0));
        auto lap_lo = call<lap_function>::with(eval, in(0, 0));
        auto flx = lap_hi - lap_lo;
        eval(out()) = flx * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0 : flx;
    }
};

struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 2>>;

    using arg_list = make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto lap_hi = call<lap_function>::with(eval, in(0, 1));
        auto lap_lo = call<lap_function>::with(eval, in(0, 0));
        auto fly = lap_hi - lap_lo;
        eval(out()) = fly * (eval(in(0, 1)) - eval(in(0, 0))) > 0 ? 0 : fly;
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-2, 2, -2, 2>>;
    using coeff = in_accessor<2>;

    using arg_list = make_arg_list<out, in, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto flx_hi = call<flx_function>::with(eval, in(0, 0));
        auto flx_lo = call<flx_function>::with(eval, in(-1, 0));

        auto fly_hi = call<fly_function>::with(eval, in(0, 0));
        auto fly_lo = call<fly_function>::with(eval, in(0, -1));

        eval(out()) = eval(in()) - eval(coeff()) * (flx_hi - flx_lo + fly_hi - fly_lo);
    }
};

using horizontal_diffusion_fused = regression_fixture<2>;

TEST_F(horizontal_diffusion_fused, test) {
    arg<0> p_coeff;
    arg<1> p_in;
    arg<2> p_out;

    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = make_computation(p_in = make_storage(repo.in),
        p_out = out,
        p_coeff = make_storage(repo.coeff),
        make_multistage(enumtype::execute<enumtype::parallel>(), make_stage<out_function>(p_out, p_in, p_coeff)));

    comp.run();
    verify(make_storage(repo.out), out);
    benchmark(comp);
}
