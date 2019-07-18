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

template <uint_t>
struct functor_copy {
    using out = inout_accessor<0, enumtype::cells>;
    using in = in_accessor<1, enumtype::cells>;
    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(in{});
    }
};

using copy_stencil_icosahedral = regression_fixture<>;

TEST_F(copy_stencil_icosahedral, test) {
    arg<0, cells> p_out;
    arg<1, cells> p_in;
    auto in = make_storage<cells>([](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; });
    auto out = make_storage<cells>();
    make_computation(
        p_out = out, p_in = in, make_multistage(execute::parallel(), make_stage<functor_copy, cells>(p_out, p_in)))
        .run();
    verify(in, out);
}
