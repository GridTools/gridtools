/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * This shows an example on how to use on_edges syntax with multiple input fields
 * (with location type edge) that are needed in the reduction over the edges of a cell
 * An typical operator that needs this functionality is the divergence where we need
 * sum_reduce(edges) {sign_edge * lengh_edge}
 * The sign of the edge indicates whether flows go inward or outward (with respect the center of the cell).
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;

struct test_on_edges_functor {
    using in1 = in_accessor<0, enumtype::edges, extent<1, -1, 1, -1>>;
    using in2 = in_accessor<1, enumtype::edges, extent<1, -1, 1, -1>>;
    using out = inout_accessor<2, enumtype::edges>;
    using param_list = make_param_list<in1, in2, out>;
    using location = enumtype::edges;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(
            on_edges([](float_type in1, float_type in2, float_type res) { return in1 + in2 * float_type{.1} + res; },
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
    auto comp = [&] {
        compute(p_in1 = make_storage<edges>(in1),
            p_in2 = make_storage<edges>(in2),
            p_out = out,
            make_multistage(execute::forward(), make_stage<test_on_edges_functor>(p_in1, p_in2, p_out)));
    };
    comp();
    verify(make_storage<edges>(ref), out);
    benchmark(comp);
}
