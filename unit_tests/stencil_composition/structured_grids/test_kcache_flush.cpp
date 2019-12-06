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
#include <gridtools/tools/cartesian_fixture.hpp>

using namespace gridtools;
using namespace cartesian;

using axis_t = gridtools::axis<3, gridtools::axis_config::offset_limit<3>, gridtools::axis_config::extra_offsets<1>>;
using kfull = axis_t::full_interval;

double in(int i, int j, int k) { return i + j + k + 1; };

struct test_kcache_flush : computation_fixture<0, axis_t> {
    test_kcache_flush() : test_kcache_flush::computation_fixture(6, 6, 2, 6, 2) {}
};

struct shift_acc_forward_flush {
    using in = in_accessor<0>;
    using out = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(out()) = eval(in());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, 0>) {
        eval(out()) = eval(out(0, 0, -1)) + eval(in());
    }
};

TEST_F(test_kcache_flush, forward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        return execute_forward().k_cached(cache_io_policy::flush(), out).stage(shift_acc_forward_flush(), in, out);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 0) = in(i, j, 0);
            for (int k = 1; k < 10; ++k)
                refv(i, j, k) = refv(i, j, k - 1) + in(i, j, k);
        }
    verify(ref, out);
}

struct shift_acc_backward_flush {
    using in = in_accessor<0>;
    using out = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(out()) = eval(in());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<0, -1>) {
        eval(out()) = eval(out(0, 0, 1)) + eval(in());
    }
};

TEST_F(test_kcache_flush, backward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        return execute_backward().k_cached(cache_io_policy::flush(), out).stage(shift_acc_backward_flush(), in, out);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 9) = in(i, j, 9);
            for (int k = 8; k >= 0; --k)
                refv(i, j, k) = refv(i, j, k + 1) + in(i, j, k);
        }
    verify(ref, out);
}
