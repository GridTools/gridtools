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

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/tools/cartesian_fixture.hpp>

using namespace gridtools;
using namespace cartesian;

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
        run(
            [](auto in, auto out) {
                GT_DECLARE_TMP(float_type, tmp);
                return execute_parallel().stage(test_functor(), in, tmp).stage(test_functor(), tmp, out);
            },
            backend_t(),
            make_grid(),
            in,
            out);
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
