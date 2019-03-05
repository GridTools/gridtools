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

#include <boost/fusion/include/make_vector.hpp>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::intent;
using gridtools::level;
using gridtools::tmp_arg;
using gridtools::uint_t;

namespace {

    template <typename Axis>
    struct parallel_functor {
        typedef accessor<0> in;
        typedef accessor<1, intent::inout> out;
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
        typedef accessor<0> in;
        typedef accessor<1, intent::inout> out;
        typedef gridtools::make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, typename Axis::template get_interval<1>) {
            eval(out()) = eval(in());
        }
    };
} // namespace

template <typename Axis>
void run_test() {

    constexpr uint_t d1 = 7;
    constexpr uint_t d2 = 8;
    constexpr uint_t d3_l = 14;
    constexpr uint_t d3_u = 16;

    using storage_info_t = typename backend_t::storage_traits_t::storage_info_t<1, 3, gridtools::halo<0, 0, 0>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<double, storage_info_t>;

    storage_info_t storage_info(d1, d2, d3_l + d3_u);

    storage_t in(storage_info, [](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); });
    storage_t out(storage_info, (double)1.5);

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto grid = gridtools::make_grid(d1, d2, Axis(d3_l, d3_u));

    auto comp = gridtools::make_computation<backend_t>(grid,
        p_in() = in,
        p_out() = out,
        gridtools::make_multistage(
            gridtools::execute::parallel(), gridtools::make_stage<parallel_functor<Axis>>(p_in(), p_out())));

    comp.run();

    comp.sync_bound_data_stores();

    auto outv = make_host_view(out);
    auto inv = make_host_view(in);
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

    using storage_info_t = typename backend_t::storage_traits_t::storage_info_t<1, 3, gridtools::halo<0, 0, 0>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<double, storage_info_t>;

    storage_info_t storage_info(d1, d2, d3_l + d3_u);

    storage_t in(storage_info, [](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); });
    storage_t out(storage_info, (double)1.5);

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;
    typedef tmp_arg<2, storage_t> p_tmp;

    auto grid = gridtools::make_grid(d1, d2, Axis(d3_l, d3_u));

    auto comp = gridtools::make_computation<backend_t>(grid,
        p_in() = in,
        p_out() = out,
        gridtools::make_multistage(gridtools::execute::parallel(),
            gridtools::make_stage<parallel_functor<Axis>>(p_in(), p_tmp()),
            gridtools::make_stage<parallel_functor<Axis>>(p_tmp(), p_out())));

    comp.run();

    comp.sync_bound_data_stores();

    auto outv = make_host_view(out);
    auto inv = make_host_view(in);
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3_l; ++k)
                EXPECT_EQ(inv(i, j, k), outv(i, j, k));
            for (int k = d3_l; k < d3_u; ++k)
                EXPECT_EQ(4 * inv(i, j, k), outv(i, j, k));
        }
}

TEST(structured_grid, kparallel) { //
    run_test<gridtools::axis<2>>();
}

TEST(structured_grid, kparallel_with_extentoffsets_around_interval) { run_test<gridtools::axis<2, 3, 5>>(); }

TEST(structured_grid, kparallel_with_temporary) { //
    run_test_with_temporary<gridtools::axis<2>>();
}

TEST(structured_grid, kparallel_with_extentoffsets_around_interval_and_temporary) {
    run_test_with_temporary<gridtools::axis<2, 3, 5>>();
}

TEST(structured_grid, kparallel_with_unused_intervals) {
    using Axis = gridtools::axis<3>;

    constexpr uint_t d1 = 7;
    constexpr uint_t d2 = 8;
    constexpr uint_t d3_1 = 14;
    constexpr uint_t d3_2 = 16;
    constexpr uint_t d3_3 = 18;

    using storage_info_t = typename backend_t::storage_traits_t::storage_info_t<1, 3, gridtools::halo<0, 0, 0>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<double, storage_info_t>;

    storage_info_t storage_info(d1, d2, d3_1 + d3_2 + d3_3);

    storage_t in(storage_info, [](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); });
    storage_t out(storage_info, (double)1.5);

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto grid = gridtools::make_grid(d1, d2, Axis(d3_1, d3_2, d3_3));

    auto comp = gridtools::make_computation<backend_t>(grid,
        p_in() = in,
        p_out() = out,
        gridtools::make_multistage(gridtools::execute::parallel(),
            gridtools::make_stage<parallel_functor_on_upper_interval<Axis>>(p_in(), p_out())));

    comp.run();

    comp.sync_bound_data_stores();

    auto outv = make_host_view(out);
    auto inv = make_host_view(in);
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
