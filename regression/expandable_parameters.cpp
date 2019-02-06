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

struct copy_functor {
    using parameters_out = inout_accessor<0>;
    using parameters_in = accessor<1>;

    using arg_list = make_arg_list<parameters_out, parameters_in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(parameters_out{}) = eval(parameters_in{});
    }
};

using expandable_parameters = regression_fixture<>;

TEST_F(expandable_parameters, test) {
    using storages_t = std::vector<storage_type>;
    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    storages_t in = {make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    arg<0, storages_t> p_out;
    arg<1, storages_t> p_in;
    tmp_arg<2, storages_t> p_tmp;

    gridtools::make_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<cache_type::IJ, cache_io_policy::local>(p_tmp)),
            make_stage<copy_functor>(p_tmp, p_in),
            make_stage<copy_functor>(p_out, p_tmp)))
        .run();
    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
