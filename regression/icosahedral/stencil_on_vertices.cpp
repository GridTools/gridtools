/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/common/binops.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;

template <uint_t>
struct test_on_vertices_functor {
    using in = in_accessor<0, enumtype::vertices, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<1, enumtype::vertices>;
    using param_list = make_param_list<in, out>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(on_vertices(binop::sum{}, float_type{}, in{}));
    }
};

using stencil_on_vertices = regression_fixture<1>;

TEST_F(stencil_on_vertices, test) {
    auto in = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };
    auto ref = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res = {};
        for (auto &&item : neighbours_of<vertices, vertices>(i, c, j, k))
            res += item.call(in);
        return res;
    };
    arg<0, vertices> p_in;
    arg<1, vertices> p_out;
    auto out = make_storage<vertices>();
    make_computation(p_in = make_storage<vertices>(in),
        p_out = out,
        make_multistage(execute::forward(), make_stage<test_on_vertices_functor, topology_t, vertices>(p_in, p_out)))
        .run();
    verify(make_storage<vertices>(ref), out);
}
