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

using namespace gridtools;

struct functor {
    using out = inout_accessor<0>;
    using param_list = make_param_list<out>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval.i() + eval.j() + eval.k();
    }
};

using positional_stencil = regression_fixture<>;

TEST_F(positional_stencil, test) {
    auto out = make_storage();

    make_positional_computation<backend_t>(
        make_grid(), p_0 = out, make_multistage(execute::forward(), make_stage<functor>(p_0)))
        .run();

    verify(make_storage([](int i, int j, int k) { return i + j + k; }), out);
}
