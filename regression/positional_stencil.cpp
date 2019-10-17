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

#include <gridtools/stencil_composition/positional.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct functor {
    using out = inout_accessor<0>;
    using i_pos = in_accessor<1>;
    using j_pos = in_accessor<2>;
    using k_pos = in_accessor<3>;
    using param_list = make_param_list<out, i_pos, j_pos, k_pos>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(i_pos()) + eval(j_pos()) + eval(k_pos());
    }
};

using positional_stencil = regression_fixture<>;

TEST_F(positional_stencil, test) {
    auto out = make_storage();

    compute(p_0 = out,
        p_1 = positional<dim::i>(),
        p_2 = positional<dim::j>(),
        p_3 = positional<dim::k>(),
        make_multistage(execute::forward(), make_stage<functor>(p_0, p_1, p_2, p_3)));

    verify(make_storage([](int i, int j, int k) { return i + j + k; }), out);
}
