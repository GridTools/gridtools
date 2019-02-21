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

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct lap {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;
    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = 4 * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

using laplacian = regression_fixture<1>;

TEST_F(laplacian, test) {
    auto in = [](int_t, int_t, int_t) { return -1.; };
    auto ref = [in](int_t i, int_t j, int_t k) {
        return 4 * in(i, j, k) - (in(i + 1, j, k) + in(i, j + 1, k) + in(i - 1, j, k) + in(i, j - 1, k));
    };
    auto out = make_storage(-7.3);

    make_computation(p_0 = out, p_1 = make_storage(in), make_multistage(execute::forward(), make_stage<lap>(p_0, p_1)))
        .run();

    verify(make_storage(ref), out);
}
