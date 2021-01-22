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

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct HorizontalExecution_142 {
        using tmp_out_field_ParAssignStmt_127 = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using in_field = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<tmp_out_field_ParAssignStmt_127, in_field>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<0, 1, 11>,
                gridtools::stencil::core::level<0, 6, 11>>) {
            eval(tmp_out_field_ParAssignStmt_127(0, 0, 0)) = eval(in_field(0, 0, 0));
        }
    };

    struct HorizontalExecution_147 {
        using out_field = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using tmp_out_field_ParAssignStmt_127 = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<out_field, tmp_out_field_ParAssignStmt_127>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<0, 1, 11>,
                gridtools::stencil::core::level<0, 6, 11>>) {
            eval(out_field(0, 0, 0)) = eval(tmp_out_field_ParAssignStmt_127(0, 0, 0));
        }
    };

    struct HorizontalExecution_158 {
        using tmp_out_field_ParAssignStmt_130 = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using in_field = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<tmp_out_field_ParAssignStmt_130, in_field>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<0, 7, 11>,
                gridtools::stencil::core::level<1, -11, 11>>) {
            eval(tmp_out_field_ParAssignStmt_130(0, 0, 0)) =
                (eval(in_field(0, 0, 0)) + static_cast<double>(static_cast<long long>(1)));
        }
    };

    struct HorizontalExecution_163 {
        using out_field = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using tmp_out_field_ParAssignStmt_130 = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<out_field, tmp_out_field_ParAssignStmt_130>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<0, 7, 11>,
                gridtools::stencil::core::level<1, -11, 11>>) {
            eval(out_field(0, 0, 0)) = eval(tmp_out_field_ParAssignStmt_130(0, 0, 0));
        }
    };

    struct HorizontalExecution_171 {
        using tmp_out_field_ParAssignStmt_133 = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using in_field = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<tmp_out_field_ParAssignStmt_133, in_field>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<1, -10, 11>,
                gridtools::stencil::core::level<1, -1, 11>>) {
            eval(tmp_out_field_ParAssignStmt_133(0, 0, 0)) = eval(in_field(0, 0, 0));
        }
    };

    struct HorizontalExecution_176 {
        using out_field = inout_accessor<0, extent<0, 0, 0, 0, 0, 0>>;
        using tmp_out_field_ParAssignStmt_133 = in_accessor<1, extent<0, 0, 0, 0, 0, 0>>;

        using param_list = make_param_list<out_field, tmp_out_field_ParAssignStmt_133>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval,
            gridtools::stencil::core::interval<gridtools::stencil::core::level<1, -10, 11>,
                gridtools::stencil::core::level<1, -1, 11>>) {
            eval(out_field(0, 0, 0)) = eval(tmp_out_field_ParAssignStmt_133(0, 0, 0));
        }
    };

    using axis_t = axis<1, axis_config::offset_limit<11>>;
    using env_t = test_environment<0, axis_t>;

    GT_REGRESSION_TEST(large_k_interval, env_t, stencil_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out_field = TypeParam::make_storage();
        auto comp = [&out_field, grid = TypeParam::make_grid(), in_field = TypeParam::make_const_storage(in)] {
            auto spec = [](auto in_field, auto out_field) {
                GT_DECLARE_TMP(double, tmp_out_field_ParAssignStmt_127);
                GT_DECLARE_TMP(double, tmp_out_field_ParAssignStmt_130);
                GT_DECLARE_TMP(double, tmp_out_field_ParAssignStmt_133);
                return multi_pass(execute_parallel()
                                      .stage(HorizontalExecution_142(), tmp_out_field_ParAssignStmt_127, in_field)
                                      .stage(HorizontalExecution_147(), out_field, tmp_out_field_ParAssignStmt_127),
                    execute_parallel()
                        .stage(HorizontalExecution_158(), tmp_out_field_ParAssignStmt_130, in_field)
                        .stage(HorizontalExecution_163(), out_field, tmp_out_field_ParAssignStmt_130),
                    execute_parallel()
                        .stage(HorizontalExecution_171(), tmp_out_field_ParAssignStmt_133, in_field)
                        .stage(HorizontalExecution_176(), out_field, tmp_out_field_ParAssignStmt_133));
            };
            run(spec, stencil_backend_t(), grid, in_field, out_field);
        };
        comp();
    };
} // namespace
