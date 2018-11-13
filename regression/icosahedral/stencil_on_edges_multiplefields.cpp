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
/*
 * This shows an example on how to use on_edges syntax with multiple input fields
 * (with location type edge) that are needed in the reduction over the edges of a cell
 * An typical operator that needs this functionality is the divergence where we need
 * sum_reduce(edges) {sign_edge * lengh_edge}
 * The sign of the edge indicates whether flows go inward or outward (with respect the center of the cell).
 */

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "unstructured_grid.hpp"

using namespace gridtools;

template <uint_t>
struct test_on_edges_functor {
    using in1 = in_accessor<0, enumtype::edges, extent<1, -1, 1, -1>>;
    using in2 = in_accessor<1, enumtype::edges, extent<1, -1, 1, -1>>;
    using out = inout_accessor<2, enumtype::edges>;
    using arg_list = boost::mpl::vector<in1, in2, out>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out{}) = eval(on_edges([](float_type in1, float_type in2, float_type res) { return in1 + in2 * .1 + res; },
            float_type{},
            in1{},
            in2{}));
    }
};

using stencil_on_edges_multiplefields = regression_fixture<1>;

TEST_F(stencil_on_edges_multiplefields, test) {
    auto in1 = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };
    auto in2 = [](int_t i, int_t c, int_t j, int_t k) { return i / 2 + c + j / 2 + k / 2; };
    auto ref = [=](int_t i, int_t c, int_t j, int_t k) {
        float_type res{};
        for (auto &&item : neighbours_of<edges, edges>(i, c, j, k))
            res += item.call(in1) + .1 * item.call(in2);
        return res;
    };
    arg<0, edges> p_in1;
    arg<1, edges> p_in2;
    arg<2, edges> p_out;
    auto out = make_storage<edges>();
    make_computation(p_in1 = make_storage<edges>(in1),
        p_in2 = make_storage<edges>(in2),
        p_out = out,
        make_multistage(enumtype::execute<enumtype::forward>(),
            make_stage<test_on_edges_functor, topology_t, edges>(p_in1, p_in2, p_out)))
        .run();
    verify(make_storage<edges>(ref), out);
}
