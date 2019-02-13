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

#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct prepare_tracers {
    using data = inout_accessor<0>;
    using data_nnow = in_accessor<1>;
    using rho = in_accessor<2>;

    using param_list = make_param_list<data, data_nnow, rho>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(data()) = eval(rho()) * eval(data_nnow());
    }
};

using advection_pdbott_prepare_tracers = regression_fixture<>;

TEST_F(advection_pdbott_prepare_tracers, test) {
    using storages_t = std::vector<storage_type>;

    arg<0, storages_t> p_out;
    arg<1, storages_t> p_in;
    arg<2, storage_type> p_rho;

    storages_t in, out;

    for (size_t i = 0; i < 11; ++i) {
        out.push_back(make_storage());
        in.push_back(make_storage(1. * i));
    }

    auto comp = gridtools::make_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        p_rho = make_storage(1.1),
        make_multistage(enumtype::execute<enumtype::forward>(), make_stage<prepare_tracers>(p_out, p_in, p_rho)));

    comp.run();
    for (size_t i = 0; i != out.size(); ++i)
        verify(make_storage([i](int_t, int_t, int_t) { return 1.1 * i; }), out[i]);

    benchmark(comp);
}
