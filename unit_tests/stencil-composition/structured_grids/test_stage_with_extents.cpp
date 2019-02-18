/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define GT_PEDANTIC_DISABLED // too stringent for this test

#include <iostream>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/structured_grids/accessor.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace test_iterate_domain {
    using namespace gridtools;

    // This is the definition of the special regions in the "vertical" direction
    struct stage1 {
        typedef accessor<0, intent::in, extent<42, 42, 42, 42>, 6> in;
        typedef accessor<1, intent::inout, extent<>, 4> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {}
    };

    struct stage2 {
        typedef accessor<0, intent::in, extent<42, 42, 42, 42>, 6> in;
        typedef accessor<1, intent::inout, extent<>, 4> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {}
    };
} // namespace test_iterate_domain

TEST(testdomain, iterate_domain_with_extents) {
    using namespace test_iterate_domain;
    typedef backend<target::x86, grid_type::structured, strategy::naive> backend_t;

    typedef storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
    typedef storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t> data_store_t;

    typedef arg<0, data_store_t> p_in;
    typedef arg<1, data_store_t> p_out;

    halo_descriptor di = {0, 0, 0, 2, 5};
    halo_descriptor dj = {0, 0, 0, 2, 5};

    auto grid = gridtools::make_grid(di, dj, 3);
    {
        auto mss_ =
            make_multistage(execute::forward(), make_stage_with_extent<stage1, extent<0, 1, 0, 0>>(p_in(), p_out()));
        auto computation_ = make_computation<backend<target::x86, grid_type_t, strategy::naive>>(grid, mss_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same<intermediate_t::extent_map_t, boost::mpl::void_>::value, "extent computation happened");
    }
    {
        auto mss_ = make_multistage(execute::forward(),
            make_stage_with_extent<stage1, extent<0, 1, 0, 0>>(p_in(), p_out()),
            make_stage_with_extent<stage2, extent<0, 1, -1, 2>>(p_out(), p_in()));
        auto computation_ = make_computation<backend<target::x86, grid_type_t, strategy::naive>>(grid, mss_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same<intermediate_t::extent_map_t, boost::mpl::void_>::value, "extent computation happened");
    }
    {
        auto mss1_ = make_multistage(execute::forward(),
            make_stage_with_extent<stage1, extent<0, 1, 0, 0>>(p_in(), p_out()),
            make_stage_with_extent<stage2, extent<0, 1, -1, 2>>(p_out(), p_in()));

        auto mss2_ = make_multistage(execute::forward(),
            make_stage_with_extent<stage1, extent<-2, 1, 0, 0>>(p_in(), p_out()),
            make_stage_with_extent<stage2, extent<-2, 1, -1, 2>>(p_out(), p_in()));

        auto computation_ = make_computation<backend<target::x86, grid_type_t, strategy::naive>>(grid, mss1_, mss2_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same<intermediate_t::extent_map_t, boost::mpl::void_>::value, "extent computation happened");
    }
    {
        auto mss1_ = make_multistage(execute::forward(),
            make_independent(make_stage_with_extent<stage1, extent<0, 1, 0, 0>>(p_in(), p_out()),
                make_stage_with_extent<stage2, extent<0, 1, -1, 2>>(p_out(), p_in())));

        auto mss2_ = make_multistage(execute::forward(),
            make_stage_with_extent<stage1, extent<-2, 1, 0, 0>>(p_in(), p_out()),
            make_stage_with_extent<stage2, extent<-2, 1, -1, 2>>(p_out(), p_in()));

        auto computation_ = make_computation<backend<target::x86, grid_type_t, strategy::naive>>(grid, mss1_, mss2_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same<intermediate_t::extent_map_t, boost::mpl::void_>::value, "extent computation happened");
    }
}
