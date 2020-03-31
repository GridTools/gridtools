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

using axis_t = gridtools::axis<3, gridtools::axis_config::offset_limit<3>>;
using kfull = axis_t::full_interval;

double in(int i, int j, int k) { return i + j + k + 1; };

struct test_kcache_fill_and_flush : computation_fixture<0, axis_t> {
    test_kcache_fill_and_flush() : test_kcache_fill_and_flush::computation_fixture(6, 6, 2, 6, 2) {}
};

struct shift_acc_forward_fill_and_flush {
    using in = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
    using param_list = make_param_list<in>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, 0>) {
        eval(in()) = eval(in()) + eval(in(0, 0, -1));
    }
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(in()) = eval(in());
    }
};

TEST_F(test_kcache_fill_and_flush, forward) {
    auto field = make_storage(in);
    auto spec = [](auto in) {
        return execute_forward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
            .stage(shift_acc_forward_fill_and_flush(), in);
    };
    run(spec, backend_t(), make_grid(), field);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 0) = in(i, j, 0);
            for (int k = 1; k < k_size(); ++k)
                refv(i, j, k) = in(i, j, k) + refv(i, j, k - 1);
        }
    verify(ref, field);
}

struct shift_acc_backward_fill_and_flush {
    using in = inout_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
    using param_list = make_param_list<in>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<0, -1>) {
        eval(in()) = eval(in()) + eval(in(0, 0, 1));
    }
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(in()) = eval(in());
    }
};

TEST_F(test_kcache_fill_and_flush, backward) {
    auto field = make_storage(in);
    auto spec = [](auto in) {
        return execute_backward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
            .stage(shift_acc_backward_fill_and_flush(), in);
    };
    run(spec, backend_t(), make_grid(), field);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, k_size() - 1) = in(i, j, k_size() - 1);
            for (int k = k_size() - 2; k >= 0; --k)
                refv(i, j, k) = refv(i, j, k + 1) + in(i, j, k);
        }
    verify(ref, field);
}

struct copy_fill {
    using in = inout_accessor<0>;
    using param_list = make_param_list<in>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull) {
        eval(in()) = eval(in());
    }
};

TEST_F(test_kcache_fill_and_flush, copy_forward) {
    auto field = make_storage(in);
    auto spec = [](auto in) {
        return execute_forward().k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in).stage(copy_fill(), in);
    };
    run(spec, backend_t(), make_grid(), field);
    verify(in, field);
}

struct scale_fill {
    using in = inout_accessor<0>;
    using param_list = make_param_list<in>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull) {
        eval(in()) = 2 * eval(in());
    }
};

TEST_F(test_kcache_fill_and_flush, scale_forward) {
    auto field = make_storage(in);
    auto spec = [](auto in) {
        return execute_forward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), in)
            .stage(scale_fill(), in);
    };
    run(spec, backend_t(), make_grid(), field);
    verify([](int i, int j, int k) { return 2 * in(i, j, k); }, field);
}
