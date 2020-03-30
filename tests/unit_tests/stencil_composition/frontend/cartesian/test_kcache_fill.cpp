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

#include <cartesian_fixture.hpp>

using namespace gridtools;
using namespace cartesian;

using axis_t = axis<3, axis_config::offset_limit<3>, axis_config::extra_offsets<1>>;
using kfull = axis_t::full_interval;

double in(int i, int j, int k) { return i + j + k + 1; };

struct test_kcache_fill : computation_fixture<0, axis_t> {
    test_kcache_fill() : test_kcache_fill::computation_fixture(6, 6, 2, 6, 2) {}
};

struct shift_acc_forward_fill {
    using in = in_accessor<0, extent<0, 0, 0, 0, -1, 1>>;
    using out = inout_accessor<1>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(out()) = eval(in()) + eval(in(0, 0, 1));
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, -1>) {
        eval(out()) = eval(in(0, 0, -1)) + eval(in()) + eval(in(0, 0, 1));
    }
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(out()) = eval(in(0, 0, -1)) + eval(in());
    }
};

TEST_F(test_kcache_fill, forward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        return execute_forward().k_cached(cache_io_policy::fill(), in).stage(shift_acc_forward_fill(), in, out);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i) {
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 0) = in(i, j, 0) + in(i, j, 1);
            for (int k = 1; k < k_size() - 1; ++k)
                refv(i, j, k) = in(i, j, k - 1) + in(i, j, k) + in(i, j, k + 1);
            refv(i, j, k_size() - 1) = in(i, j, k_size() - 1) + in(i, j, k_size() - 2);
        }
    }
    verify(ref, out);
}

struct shift_acc_backward_fill {
    using in = in_accessor<0, extent<0, 0, 0, 0, -1, 1>>;
    using out = inout_accessor<1>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(out()) = eval(in()) + eval(in(0, 0, -1));
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, -1>) {
        eval(out()) = eval(in(0, 0, 1)) + eval(in()) + eval(in(0, 0, -1));
    }
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(out()) = eval(in()) + eval(in(0, 0, 1));
    }
};

TEST_F(test_kcache_fill, backward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        return execute_backward().k_cached(cache_io_policy::fill(), in).stage(shift_acc_backward_fill(), in, out);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i) {
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, k_size() - 1) = in(i, j, k_size() - 1) + in(i, j, k_size() - 2);
            for (int_t k = k_size() - 2; k >= 1; --k)
                refv(i, j, k) = in(i, j, k + 1) + in(i, j, k) + in(i, j, k - 1);
            refv(i, j, 0) = in(i, j, 1) + in(i, j, 0);
        }
    }
    verify(ref, out);
}

struct copy_fill {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;

    using param_list = make_param_list<in, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull) {
        eval(out()) = eval(in());
    }
};

TEST_F(test_kcache_fill, fill_copy_forward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        return execute_forward().k_cached(cache_io_policy::fill(), in).stage(copy_fill(), in, out);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    verify(in, out);
}
