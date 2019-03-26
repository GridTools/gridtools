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
#include <gridtools/tools/backend_select.hpp>

#include "../../test_helper.hpp"

/**
 * Compile-time test to ensure that types are correct in all call_proc stages
 */

using namespace gridtools;
using namespace gridtools::execute;
using namespace gridtools::expressions;

namespace {
    // used to ensure that types are correctly passed between function calls (no implicit conversion)
    template <typename tag>
    struct special_type {};

    struct in_tag {};
    struct out_tag {};
} // namespace

class call_proc_stress_types : public testing::Test {
  protected:
    using storage_info_t = gridtools::storage_traits<backend_t::backend_target_t>::storage_info_t<0, 3>;
    using data_store_in_t =
        gridtools::storage_traits<backend_t::backend_target_t>::data_store_t<special_type<in_tag>, storage_info_t>;
    using data_store_out_t =
        gridtools::storage_traits<backend_t::backend_target_t>::data_store_t<special_type<out_tag>, storage_info_t>;

    gridtools::grid<gridtools::axis<1>::axis_interval_t> grid;

    data_store_in_t in;
    data_store_out_t out;

    typedef arg<0, data_store_in_t> p_in;
    typedef arg<1, data_store_out_t> p_out;

    call_proc_stress_types() : grid(make_grid(1, 1, 1)), in(storage_info_t{1, 1, 1}), out(storage_info_t{1, 1, 1}) {}
};

namespace {
    struct local_tag {};

    struct triple_nesting_with_type_switching_third_stage {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> local;
        typedef make_param_list<out, local> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using local_type = decay_t<decltype(eval(local{}))>;
            (void)ASSERT_TYPE_EQ<special_type<local_tag>, local_type>{};
        }
    };

    struct triple_nesting_with_type_switching_second_stage {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using in_type = decay_t<decltype(eval(in{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in_tag>, in_type>{};

            special_type<local_tag> local{};

            call_proc<triple_nesting_with_type_switching_third_stage>::with(eval, out(), local);
        }
    };

    struct triple_nesting_with_type_switching_first_stage {
        typedef inout_accessor<0> out;
        typedef in_accessor<1> in;
        typedef make_param_list<out, in> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            using out_type = decay_t<decltype(eval(out{}))>;
            (void)ASSERT_TYPE_EQ<special_type<out_tag>, out_type>{};

            using in_type = decay_t<decltype(eval(in{}))>;
            (void)ASSERT_TYPE_EQ<special_type<in_tag>, in_type>{};

            call_proc<triple_nesting_with_type_switching_second_stage>::with(eval, in(), out());
        }
    };
} // namespace

TEST_F(call_proc_stress_types, triple_nesting_with_type_switching) {
    auto comp = gridtools::make_computation<backend_t>(grid,
        p_in{} = in,
        p_out{} = out,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<triple_nesting_with_type_switching_first_stage>(p_out(), p_in())));
    comp.run();
}
