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

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;

struct lap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;

    using arg_list = boost::mpl::vector<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) =
            float_type{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<0, 1, 0, 0>>;
    using lap = in_accessor<2, extent<0, 1, 0, 0>>;

    using arg_list = boost::mpl::vector<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(lap(1, 0)) - eval(lap(0, 0));
        if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<0, 0, 0, 1>>;
    using lap = in_accessor<2, extent<0, 0, 0, 1>>;

    using arg_list = boost::mpl::vector<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(lap(0, 1)) - eval(lap(0, 0));
        if (eval(out()) * (eval(in(0, 1)) - eval(in(0, 0))) > 0)
            eval(out()) = 0.;
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using flx = in_accessor<2, extent<-1, 0, 0, 0>>;
    using fly = in_accessor<3, extent<0, 0, -1, 0>>;
    using coeff = in_accessor<4>;

    using arg_list = boost::mpl::vector<out, in, flx, fly, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
    }
};

using HorizontalDiffusion = regression_fixture<2>;

TEST_F(HorizontalDiffusion, Test) {
    tmp_arg<0> p_lap;
    tmp_arg<1> p_flx;
    tmp_arg<2> p_fly;
    arg<3> p_coeff;
    arg<4> p_in;
    arg<5> p_out;

    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = make_computation(p_in = make_storage(repo.in),
        p_out = out,
        p_coeff = make_storage(repo.coeff),
        make_multistage(enumtype::execute<enumtype::parallel, 20>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_lap, p_flx, p_fly)),
            make_stage<lap_function>(p_lap, p_in),
            make_independent(
                make_stage<flx_function>(p_flx, p_in, p_lap), make_stage<fly_function>(p_fly, p_in, p_lap)),
            make_stage<out_function>(p_out, p_in, p_flx, p_fly, p_coeff)));

    comp.run();
    verify(make_storage(repo.out), out);
    benchmark(comp);
}
