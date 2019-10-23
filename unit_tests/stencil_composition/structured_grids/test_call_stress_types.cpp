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

#include <gridtools/meta/type_traits.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

#include "../../test_helper.hpp"

/**
 * Compile-time test to ensure that types are correct in all call stages
 */

using namespace gridtools;
using namespace execute;
using namespace expressions;

namespace {
    // used to ensure that types are correctly passed between function calls (no implicit conversion)
    template <typename tag>
    struct special_type {};

    struct in1_tag {};
    struct in2_tag {};
    struct out_tag {};
} // namespace

class call_stress_types : public testing::Test {
  protected:
    using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3>;
    using data_store_in1_t = storage_traits<backend_t>::data_store_t<special_type<in1_tag>, storage_info_t>;
    using data_store_in2_t = storage_traits<backend_t>::data_store_t<special_type<in2_tag>, storage_info_t>;
    using data_store_out_t = storage_traits<backend_t>::data_store_t<special_type<out_tag>, storage_info_t>;

    gridtools::grid<axis<1>::axis_interval_t> grid;

    data_store_in1_t in1;
    data_store_in2_t in2;
    data_store_out_t out;

    call_stress_types()
        : grid(make_grid(1, 1, 1)), in1(storage_info_t{1, 1, 1}), in2(storage_info_t{1, 1, 1}),
          out(storage_info_t{1, 1, 1}) {}
};

namespace {
    struct forced_tag {};

    struct simple_callee_with_forced_return_type {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<forced_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};
        }
    };

    struct simple_caller_with_forced_return_type {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            auto result =
                call<simple_callee_with_forced_return_type>::return_type<special_type<forced_tag>>::with(eval, in{});

            using result_type = decltype(result);
            (void)ASSERT_TYPE_EQ<special_type<forced_tag>, result_type>{};
        }
    };
} // namespace

TEST_F(call_stress_types, simple_force_return_type) {
    easy_run(simple_caller_with_forced_return_type(), backend_t(), grid, in1, out);
}

namespace {
    struct simple_callee_with_deduced_return_type {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};
        }
    };

    struct simple_caller_with_deduced_return_type {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            auto result = call<simple_callee_with_deduced_return_type>::with(eval, in{});

            using result_type = decltype(result);
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, result_type>{};
        }
    };
} // namespace

TEST_F(call_stress_types, simple_deduced_return_type) {
    easy_run(simple_caller_with_deduced_return_type(), backend_t(), grid, in1, out);
}

namespace {
    struct local_tag {};

    struct triple_nesting_with_type_switching_third_stage {
        typedef in_accessor<0> in2;
        typedef in_accessor<1> local;
        typedef inout_accessor<2> out;
        typedef in_accessor<3> in1;
        typedef make_param_list<in2, local, out, in1> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            // the new convention is that the return type (here "out) is deduced from the first argument in the call
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in1{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};

            using in2_type = std::decay_t<decltype(eval(in2{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, in2_type>{};

            using local_type = std::decay_t<decltype(eval(local{}))>;
            (void)ASSERT_TYPE_EQ<special_type<local_tag>, local_type>{};
        }
    };

    struct triple_nesting_with_type_switching_second_stage {
        typedef in_accessor<0> in1;
        typedef inout_accessor<1> out;
        typedef in_accessor<2> in2;
        typedef make_param_list<in1, out, in2> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            // the expected type differs here in "call" vs "call_proc"
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in1{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};

            using in2_type = std::decay_t<decltype(eval(in2{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, in2_type>{};

            special_type<local_tag> local{};

            auto result = call<triple_nesting_with_type_switching_third_stage>::with(eval, in2(), local, in1());
            using result_type = decltype(result);
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, result_type>{};
        }
    };

    struct triple_nesting_with_type_switching_first_stage {
        typedef in_accessor<0> in1;
        typedef inout_accessor<1> out;
        typedef in_accessor<2> in2;
        typedef make_param_list<in1, out, in2> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in1{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};

            using in2_type = std::decay_t<decltype(eval(in2{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, in2_type>{};

            auto result = call<triple_nesting_with_type_switching_second_stage>::with(eval, in1(), in2());
            using result_type = decltype(result);
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, result_type>{};
        }
    };
} // namespace

TEST_F(call_stress_types, triple_nesting_with_type_switching) {
    easy_run(triple_nesting_with_type_switching_first_stage(), backend_t(), grid, in1, out, in2);
}

namespace {
    struct triple_nesting_with_type_switching_and_call_proc_second_stage {
        typedef in_accessor<0> in1;
        typedef inout_accessor<1> out;
        typedef in_accessor<2> in2;
        typedef make_param_list<in1, out, in2> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            // in contrast to the example where this is stage is called from "call" (not "call_proc")
            // the type here is different!
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in1{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};

            using in2_type = std::decay_t<decltype(eval(in2{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, in2_type>{};

            special_type<local_tag> local{};

            auto result = call<triple_nesting_with_type_switching_third_stage>::with(eval, in2(), local, in1());
            using result_type = decltype(result);
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, result_type>{};
        }
    };

    struct triple_nesting_with_type_switching_and_call_proc_first_stage {
        typedef in_accessor<0> in1;
        typedef inout_accessor<1> out;
        typedef in_accessor<2> in2;
        typedef make_param_list<in1, out, in2> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = std::decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using in1_type = std::decay_t<decltype(eval(in1{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in1_tag>, in1_type>{};

            using in2_type = std::decay_t<decltype(eval(in2{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in2_tag>, in2_type>{};

            call_proc<triple_nesting_with_type_switching_and_call_proc_second_stage>::with(eval, in1(), out(), in2());
        }
    };
} // namespace

TEST_F(call_stress_types, triple_nesting_with_type_switching_and_call_proc) {
    easy_run(triple_nesting_with_type_switching_and_call_proc_first_stage(), backend_t(), grid, in1, out, in2);
}
