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

using axis_t = axis<3, axis_config::offset_limit<3>, axis_config::extra_offsets<1>>;
using kfull = axis_t::full_interval;

double in(int i, int j, int k) { return i + j + k + 1; };

struct test_kcache_local : computation_fixture<0, axis_t> {
    test_kcache_local() : test_kcache_local::computation_fixture(6, 6, 2, 6, 2) {}
};

struct shift_acc_forward {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using buff = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<in, out, buff>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, 0>) {
        eval(buff()) = eval(buff(0, 0, -1)) + eval(in());
        eval(out()) = eval(buff());
    }
};

TEST_F(test_kcache_local, forward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        GT_DECLARE_TMP(float_type, tmp);
        return execute_forward().k_cached(tmp).stage(shift_acc_forward(), in, out, tmp);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 0) = in(i, j, 0);
            for (int k = 1; k < 10; ++k) {
                refv(i, j, k) = refv(i, j, k - 1) + in(i, j, k);
            }
        }
    verify(ref, out);
}

struct shift_acc_backward {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using buff = inout_accessor<2, extent<0, 0, 0, 0, 0, 1>>;

    using param_list = make_param_list<in, out, buff>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<0, -1>) {
        eval(buff()) = eval(buff(0, 0, 1)) + eval(in());
        eval(out()) = eval(buff());
    }
};

TEST_F(test_kcache_local, backward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        GT_DECLARE_TMP(float_type, tmp);
        return execute_backward().k_cached(tmp).stage(shift_acc_backward(), in, out, tmp);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            refv(i, j, 9) = in(i, j, 9);
            for (int k = 8; k >= 0; --k) {
                refv(i, j, k) = refv(i, j, k + 1) + in(i, j, k);
            }
        }
    verify(ref, out);
}

struct biside_large_kcache_forward {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using buff = inout_accessor<2, extent<0, 0, 0, 0, -2, 1>>;

    using param_list = make_param_list<in, out, buff>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, 1)) = eval(in()) / 2;
        eval(out()) = eval(buff());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level::shift<1>) {
        eval(buff(0, 0, 1)) = eval(in()) / 2;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) / 4;
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<2, -1>) {
        eval(buff(0, 0, 1)) = eval(in()) / 2;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) / 4 + eval(buff(0, 0, -2)) * .12;
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) / 4 + eval(buff(0, 0, -2)) * .12;
    }
};

TEST_F(test_kcache_local, biside_forward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        GT_DECLARE_TMP(float_type, tmp);
        return execute_forward().k_cached(tmp).stage(biside_large_kcache_forward(), in, out, tmp);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    auto buff = make_storage();
    auto buffv = buff->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int_t j = 0; j < d(1); ++j) {
            buffv(i, j, 0) = in(i, j, 0);
            buffv(i, j, 1) = in(i, j, 0) / 2;
            refv(i, j, 0) = in(i, j, 0);
            buffv(i, j, 2) = in(i, j, 1) / 2;
            refv(i, j, 1) = buffv(i, j, 1) + buffv(i, j, 0) / 4;
            for (int k = 2; k < 10; ++k) {
                if (k != 9)
                    buffv(i, j, k + 1) = in(i, j, k) / 2;
                refv(i, j, k) = buffv(i, j, k) + buffv(i, j, k - 1) / 4 + .12 * buffv(i, j, k - 2);
            }
        }
    verify(ref, out);
}

struct biside_large_kcache_backward {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using buff = inout_accessor<2, extent<0, 0, 0, 0, -1, 2>>;

    using param_list = make_param_list<in, out, buff>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, -1)) = eval(in()) / 2;
        eval(out()) = eval(buff());
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::last_level::shift<-1>) {
        eval(buff(0, 0, -1)) = eval(in()) / 2;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) / 4;
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::modify<1, -2>) {
        eval(buff(0, 0, -1)) = eval(in()) / 2;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) / 4 + eval(buff(0, 0, 2)) * .12;
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, kfull::first_level) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) / 4 + eval(buff(0, 0, 2)) * .12;
    }
};

TEST_F(test_kcache_local, biside_backward) {
    auto out = make_storage();
    auto spec = [](auto in, auto out) {
        GT_DECLARE_TMP(float_type, tmp);
        return execute_backward().k_cached(tmp).stage(biside_large_kcache_backward(), in, out, tmp);
    };
    run(spec, backend_t(), make_grid(), make_storage(in), out);
    auto ref = make_storage();
    auto refv = ref->host_view();
    auto buff = make_storage();
    auto buffv = buff->host_view();
    for (int i = 0; i < d(0); ++i)
        for (int j = 0; j < d(1); ++j) {
            buffv(i, j, 9) = in(i, j, 9);
            buffv(i, j, 8) = in(i, j, 9) / 2;
            refv(i, j, 9) = in(i, j, 9);
            buffv(i, j, 7) = in(i, j, 8) / 2;
            refv(i, j, 8) = buffv(i, j, 8) + buffv(i, j, 9) / 4;
            for (int_t k = 7; k >= 0; --k) {
                if (k != 0)
                    buffv(i, j, k - 1) = in(i, j, k) / 2;
                refv(i, j, k) = buffv(i, j, k) + buffv(i, j, k + 1) / 4 + .12 * buffv(i, j, k + 2);
            }
        }
    verify(ref, out);
}
