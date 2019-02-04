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
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out{}) = eval(in{});
    }
};

using expandable_parameters_icosahedral = regression_fixture<>;

TEST_F(expandable_parameters_icosahedral, test) {
    using storages_t = std::vector<storage_type<cells>>;
    storages_t in = {make_storage<cells>(10.),
        make_storage<cells>(20.),
        make_storage<cells>(30.),
        make_storage<cells>(40.),
        make_storage<cells>(50.)};
    storages_t out = {make_storage<cells>(1.),
        make_storage<cells>(2.),
        make_storage<cells>(3.),
        make_storage<cells>(4.),
        make_storage<cells>(5.)};

    arg<0, cells, storages_t> p_out;
    arg<1, cells, storages_t> p_in;

    gridtools::make_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        make_multistage(execute<execution::forward>(), make_stage<functor_copy, topology_t, cells>(p_out, p_in)))
        .run();

    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
