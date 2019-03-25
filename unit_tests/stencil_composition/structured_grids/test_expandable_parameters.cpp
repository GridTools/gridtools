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

#include <gridtools/stencil_composition/expandable_parameters/make_computation.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions/stencil_functions.hpp>
#include <gridtools/tools/computation_fixture.hpp>

using namespace gridtools;
using namespace gridtools::execute;
using namespace gridtools::expressions;

struct expandable_parameters : computation_fixture<> {
    expandable_parameters() : computation_fixture<>(13, 9, 7) {}
    using storages_t = std::vector<storage_type>;

    template <class... Args>
    void run_computation(Args &&... args) const {
        gridtools::make_expandable_computation<backend_t>(expand_factor<2>(), make_grid(), std::forward<Args>(args)...)
            .run();
    }

    void verify(storages_t const &expected, storages_t const &actual) const {
        EXPECT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i != expected.size(); ++i)
            computation_fixture<>::verify(expected[i], actual[i]);
    }
};

struct expandable_parameters_copy : expandable_parameters {

    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    storages_t in = {make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    template <class Functor>
    void run_computation() {
        arg<0, storages_t> p_out;
        arg<1, storages_t> p_in;
        expandable_parameters::run_computation(
            p_in = in, p_out = out, make_multistage(execute::forward(), make_stage<Functor>(p_out, p_in)));
    }

    ~expandable_parameters_copy() { verify(in, out); }
};

struct copy_functor {
    typedef accessor<0, intent::inout> out;
    typedef accessor<1, intent::in> in;

    typedef make_param_list<out, in> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(out{}) = eval(in{});
    }
};

TEST_F(expandable_parameters_copy, copy) { run_computation<copy_functor>(); }

struct copy_functor_with_expression {
    typedef accessor<0, intent::inout> out;
    typedef accessor<1, intent::in> in;

    typedef make_param_list<out, in> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        // use an expression which is equivalent to a copy to simplify the check
        eval(out{}) = eval(2. * in{} - in{});
    }
};

TEST_F(expandable_parameters_copy, copy_with_expression) { run_computation<copy_functor_with_expression>(); }

struct call_proc_copy_functor {
    typedef accessor<0, intent::inout> out;
    typedef accessor<1, intent::in> in;

    typedef make_param_list<out, in> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        call_proc<copy_functor>::with(eval, out(), in());
    }
};

TEST_F(expandable_parameters_copy, call_proc_copy) { run_computation<call_proc_copy_functor>(); }

struct call_copy_functor {
    typedef accessor<0, intent::inout> out;
    typedef accessor<1, intent::in> in;

    typedef make_param_list<out, in> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(out()) = call<copy_functor>::with(eval, in());
    }
};

TEST_F(expandable_parameters_copy, call_copy) { run_computation<call_copy_functor>(); }

struct shift_functor {
    typedef accessor<0, intent::inout, extent<0, 0, 0, 0, -1, 0>> out;

    typedef make_param_list<out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(out()) = eval(out(0, 0, -1));
    }
};

struct call_shift_functor {
    typedef accessor<0, intent::inout, extent<0, 0, 0, 0, -1, 0>> out;

    typedef make_param_list<out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, axis<1>::full_interval::modify<1, 0>) {
        call_proc<shift_functor>::with(eval, out());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &, axis<1>::full_interval::first_level) {}
};

TEST_F(expandable_parameters, call_shift) {
    auto expected = [&](float_type value) { return make_storage([=](int_t, int_t, int_t) { return value; }); };
    auto in = [&](float_type value) {
        return make_storage([=](int_t, int_t, int_t k) { return k == 0 ? value : -1; });
    };

    storages_t actual = {in(14), in(15), in(16), in(17), in(18)};
    arg<0, storages_t> plh;
    run_computation(plh = actual, make_multistage(execute::forward(), make_stage<call_shift_functor>(plh)));
    verify({expected(14), expected(15), expected(16), expected(17), expected(18)}, actual);
}

TEST_F(expandable_parameters, caches) {
    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    auto in = make_storage(42.);

    arg<0, storages_t> p_out;
    arg<1> p_in;
    tmp_arg<1> p_tmp;
    run_computation(p_in = in,
        p_out = out,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp)),
            make_stage<copy_functor>(p_tmp, p_in),
            make_stage<copy_functor>(p_out, p_tmp)));
    verify({in, in, in, in, in}, out);
}
