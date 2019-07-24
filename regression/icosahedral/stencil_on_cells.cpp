/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;

template <uint_t>
struct test_on_cells_functor {
    using in = in_accessor<0, enumtype::cells, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<in, out>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(on_cells([](float_type lhs, float_type rhs) { return lhs + rhs; }, 0., in{}));
    }
};

using stencil_on_cells = regression_fixture<1>;

TEST_F(stencil_on_cells, test) {
    auto in = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };
    auto ref = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res = {};
        for (auto &&item : neighbours_of<cells, cells>(i, c, j, k))
            res += item.call(in);
        return res;
    };
    arg<0, cells> p_in;
    arg<1, cells> p_out;
    auto out = make_storage<cells>();
    auto comp = make_computation(p_in = make_storage<cells>(in),
        p_out = out,
        make_multistage(execute::forward(), make_stage<test_on_cells_functor, cells>(p_in, p_out)));
    comp.run();
    verify(make_storage<cells>(ref), out);
    benchmark(comp);
}
