/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

using namespace gridtools;

struct test_functor {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

struct fixture : computation_fixture<> {
    fixture() : computation_fixture<>(6, 6, 10) {}
};

TEST_F(fixture, run) {
    std::function<void(storage_type, storage_type)> comp = [&](storage_type in, storage_type out) {
        arg<0> p_in;
        arg<1> p_out;
        tmp_arg<2, float_type> p_tmp;
        compute<backend_t>(p_in = in,
            p_out = out,
            make_multistage(
                execute::forward(), make_stage<test_functor>(p_in, p_tmp), make_stage<test_functor>(p_tmp, p_out)));
    };

    auto do_test = [&](int n) {
        auto in = make_storage([=](int i, int j, int k) { return i + j + k + n; });
        auto out = make_storage();
        comp(in, out);
        verify(in, out);
    };

    do_test(3);
    do_test(7);
}
