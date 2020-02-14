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

#include <gridtools/stencil_composition/icosahedral.hpp>
#include <gridtools/tools/icosahedral_regression_fixture.hpp>

using namespace gridtools;
using namespace icosahedral;

struct functor_copy {
    using out = inout_accessor<0, cells>;
    using in = in_accessor<1, cells>;
    using param_list = make_param_list<out, in>;
    using location = cells;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in());
    }
};

using copy_stencil_icosahedral = regression_fixture<>;

TEST_F(copy_stencil_icosahedral, test) {
    auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
    auto out = make_storage<cells>();
    run_single_stage(functor_copy(), backend_t(), make_grid(), out, make_storage<cells>(in));
    verify(in, out);
}
