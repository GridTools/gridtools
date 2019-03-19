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

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in());
    }
};

struct copy_stencil : regression_fixture<0> {
    storage_type in = make_storage([](int i, int j, int k) { return i + j + k; });
    storage_type out = make_storage(-1.);
};

TEST_F(copy_stencil, test) {
    auto comp =
        make_computation(p_0 = in, p_1 = out, make_multistage(execute::parallel(), make_stage<copy_functor>(p_0, p_1)));

    comp.run();
    verify(in, out);
    benchmark(comp);
}

TEST_F(copy_stencil, with_extents) {
    make_computation(p_0 = in,
        p_1 = out,
        make_multistage(execute::parallel(), make_stage_with_extent<copy_functor, extent<>>(p_0, p_1)))
        .run();
    verify(in, out);
}
