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

struct copy_functor {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

using copy_stencil = regression_fixture<>;

TEST_F(copy_stencil, test) {
    auto in = make_storage([](int i, int j, int k) { return i + j + k; });
    auto out = make_storage(-1.);
    auto comp = [&] {
        compute(p_0 = in, p_1 = out, make_multistage(execute::parallel(), make_stage<copy_functor>(p_0, p_1)));
    };
    comp();
    verify(in, out);
    benchmark(comp);
}
