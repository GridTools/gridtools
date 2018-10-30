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

/**
  @file
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using namespace gridtools;

struct wlap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;
    using crlato = in_accessor<2>;
    using crlatu = in_accessor<3>;

    using arg_list = boost::mpl::vector<out, in, crlato, crlatu>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(in(1, 0)) + eval(in(-1, 0)) - float_type{2} * eval(in()) +
                      eval(crlato()) * (eval(in(0, 1)) - eval(in())) + eval(crlatu()) * (eval(in(0, -1)) - eval(in()));
    }
};

struct divflux_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using lap = in_accessor<2, extent<-1, 1, -1, 1>>;
    using crlato = in_accessor<3>;
    using coeff = in_accessor<4>;

    using arg_list = boost::mpl::vector<out, in, lap, crlato, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        auto fluxx = eval(lap(1, 0)) - eval(lap());
        auto fluxx_m = eval(lap()) - eval(lap(-1, 0));

        auto fluxy = eval(crlato()) * (eval(lap(0, 1)) - eval(lap()));
        auto fluxy_m = eval(crlato()) * (eval(lap()) - eval(lap(0, -1)));

        eval(out()) = eval(in()) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * eval(coeff());
    }
};

using SimpleHorizontalDiffusion = regression_fixture<2>;

TEST_F(SimpleHorizontalDiffusion, Test) {
    tmp_arg<0> p_lap;
    arg<1> p_coeff;
    arg<2> p_in;
    arg<3> p_out;
    arg<4, j_storage_type> p_crlato;
    arg<5, j_storage_type> p_crlatu;

    auto out = make_storage(0.);

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = make_computation(p_coeff = make_storage(repo.coeff),
        p_in = make_storage(repo.in),
        p_out = out,
        p_crlato = make_storage<j_storage_type>(repo.crlato),
        p_crlatu = make_storage<j_storage_type>(repo.crlatu),
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_lap)),
            make_stage<wlap_function>(p_lap, p_in, p_crlato, p_crlatu),
            make_stage<divflux_function>(p_out, p_in, p_lap, p_crlato, p_coeff)));

    comp.run();
    verify(make_storage(repo.out_simple), out);
    benchmark(comp);
}
