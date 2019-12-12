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

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/tools/cartesian_regression_fixture.hpp>

using namespace gridtools;
using namespace cartesian;

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
    auto in = [](int i, int j, int k) { return i + j + k; };
    auto out = make_storage();
    auto comp = [&out, grid = make_grid(), in = make_storage<float_type const>(in)] {
        easy_run(copy_functor(), backend_t(), grid, in, out);
    };
    comp();
    verify(in, out);
    benchmark(comp);
}
