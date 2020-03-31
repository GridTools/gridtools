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
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include <backend_select.hpp>

namespace {
    using namespace gridtools;
    using namespace cartesian;

    template <typename Axis>
    struct parallel_functor {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef gridtools::make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, typename Axis::template get_interval<0>) {
            eval(out()) = eval(in());
        }
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, typename Axis::template get_interval<1>) {
            eval(out()) = 2 * eval(in());
        }
    };

    template <typename Axis>
    struct parallel_functor_on_upper_interval {
        typedef in_accessor<0> in;
        typedef inout_accessor<1> out;
        typedef gridtools::make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, typename Axis::template get_interval<1>) {
            eval(out()) = eval(in());
        }
    };

    const auto builder = storage::builder<storage_traits_t>.type<double>();

    template <typename Axis>
    void run_test() {

        constexpr uint_t d1 = 7;
        constexpr uint_t d2 = 8;
        constexpr uint_t d3_l = 14;
        constexpr uint_t d3_u = 16;

        auto builder = ::builder.dimensions(d1, d2, d3_l + d3_u);

        auto in = builder.initializer([](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); })();
        auto out = builder.value(1.5)();

        auto grid = make_grid(d1, d2, Axis(d3_l, d3_u));

        run_single_stage(parallel_functor<Axis>(), backend_t(), grid, in, out);

        auto outv = out->host_view();
        auto inv = in->host_view();
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3_l; ++k)
                    EXPECT_EQ(inv(i, j, k), outv(i, j, k));
                for (int k = d3_l; k < d3_u; ++k)
                    EXPECT_EQ(2 * inv(i, j, k), outv(i, j, k));
            }
    }

    template <typename Axis>
    void run_test_with_temporary() {

        constexpr uint_t d1 = 7;
        constexpr uint_t d2 = 8;
        constexpr uint_t d3_l = 14;
        constexpr uint_t d3_u = 16;

        auto builder = ::builder.dimensions(d1, d2, d3_l + d3_u);

        auto in = builder.initializer([](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); })();
        auto out = builder.value(1.5)();

        auto grid = make_grid(d1, d2, Axis(d3_l, d3_u));

        run(
            [](auto in, auto out) {
                GT_DECLARE_TMP(double, tmp);
                return execute_parallel()
                    .stage(parallel_functor<Axis>(), in, tmp)
                    .stage(parallel_functor<Axis>(), tmp, out);
            },
            backend_t(),
            grid,
            in,
            out);

        auto outv = out->host_view();
        auto inv = in->host_view();
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3_l; ++k)
                    EXPECT_EQ(inv(i, j, k), outv(i, j, k));
                for (int k = d3_l; k < d3_u; ++k)
                    EXPECT_EQ(4 * inv(i, j, k), outv(i, j, k));
            }
    }

    TEST(structured_grid, kparallel) { run_test<gridtools::axis<2>>(); }

    TEST(structured_grid, kparallel_with_extentoffsets_around_interval) {
        run_test<
            gridtools::axis<2, gridtools::axis_config::offset_limit<5>, gridtools::axis_config::extra_offsets<3>>>();
    }

    TEST(structured_grid, kparallel_with_temporary) { run_test_with_temporary<gridtools::axis<2>>(); }

    TEST(structured_grid, kparallel_with_extentoffsets_around_interval_and_temporary) {
        run_test_with_temporary<
            gridtools::axis<2, gridtools::axis_config::offset_limit<5>, gridtools::axis_config::extra_offsets<3>>>();
    }

    TEST(structured_grid, kparallel_with_unused_intervals) {
        using Axis = gridtools::axis<3>;

        constexpr int_t d1 = 7;
        constexpr int_t d2 = 8;
        constexpr int_t d3_1 = 14;
        constexpr int_t d3_2 = 16;
        constexpr int_t d3_3 = 18;

        auto builder = ::builder.dimensions(d1, d2, d3_1 + d3_2 + d3_3);

        auto in = builder.initializer([](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); })();
        auto out = builder.value(1.5)();

        auto grid = make_grid(d1, d2, Axis(d3_1, d3_2, d3_3));

        run_single_stage(parallel_functor_on_upper_interval<Axis>(), backend_t(), grid, in, out);

        auto outv = out->host_view();
        auto inv = in->host_view();
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3_1; ++k)
                    EXPECT_EQ(1.5, outv(i, j, k));
                for (int k = d3_1; k < d3_1 + d3_2; ++k)
                    EXPECT_EQ(inv(i, j, k), outv(i, j, k));
                for (int k = d3_1 + d3_2; k < d3_1 + d3_2 + d3_3; ++k)
                    EXPECT_EQ(1.5, outv(i, j, k));
            }
    }
} // namespace
