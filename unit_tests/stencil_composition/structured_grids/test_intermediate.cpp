/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace test_intermediate {
    using namespace gridtools;

    struct stage1 {
        using in1 = accessor<0, intent::in, extent<0, 1, -1, 0, 0, 1>>;
        using in2 = accessor<1, intent::in, extent<0, 1, -1, 0, -1, 1>>;
        using out = accessor<2, intent::inout, extent<>>;
        using param_list = make_param_list<in1, in2, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &) {}
    };

    struct stage2 {
        using in1 = accessor<0, intent::in, extent<-1, 0, 0, 1, -1, 0>>;
        using in2 = accessor<1, intent::in, extent<-1, 1, -1, 0, -1, 1>>;
        using out = accessor<2, intent::inout, extent<>>;
        using param_list = make_param_list<in1, in2, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &) {}
    };
} // namespace test_intermediate

TEST(intermediate, test_get_arg_functions) {
    using namespace test_intermediate;

    using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 1>;
    using data_store_t = storage_traits<backend_t>::data_store_t<float_type, storage_info_t>;

    using p_in1 = arg<0, data_store_t>;
    using p_in2 = arg<1, data_store_t>;
    using p_tmp1 = arg<2, data_store_t>;
    using p_tmp2 = arg<3, data_store_t>;
    using p_tmp3 = arg<4, data_store_t>;
    using p_out = arg<5, data_store_t>;

    halo_descriptor di = {3, 3, 3, 3, 7};
    halo_descriptor dj = {3, 3, 3, 3, 7};

    auto grid = gridtools::make_grid(di, dj, 3);
    {
        auto mss_ = make_multistage(execute::forward(), make_stage<stage1>(p_in1(), p_in2(), p_out()));
        computation<p_in1, p_in2, p_out> comp = make_computation<backend_t>(grid, mss_);

        EXPECT_EQ((rt_extent{0, 1, -1, 0, 0, 1}), comp.get_arg_extent(p_in1()));
        EXPECT_EQ((rt_extent{0, 1, -1, 0, -1, 1}), comp.get_arg_extent(p_in2()));
        EXPECT_EQ((rt_extent{0, 0, 0, 0, 0, 0}), comp.get_arg_extent(p_out()));

        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_out()));
    }

    {
        auto mss_ = make_multistage(execute::forward(),
            make_stage<stage1>(p_in1(), p_in2(), p_tmp1()),
            make_stage<stage2>(p_in1(), p_tmp1(), p_out()));
        computation<p_in1, p_in2, p_tmp1, p_out> comp = make_computation<backend_t>(grid, mss_);

        EXPECT_EQ((rt_extent{-1, 2, -2, 1, -1, 2}), comp.get_arg_extent(p_in1()));
        EXPECT_EQ((rt_extent{-1, 2, -2, 0, -2, 2}), comp.get_arg_extent(p_in2()));
        EXPECT_EQ((rt_extent{-1, 1, -1, 0, -1, 1}), comp.get_arg_extent(p_tmp1()));
        EXPECT_EQ((rt_extent{0, 0, 0, 0, 0, 0}), comp.get_arg_extent(p_out()));

        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_tmp1()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_out()));
    }

    {
        auto mss_ = make_multistage(execute::forward(),
            make_stage<stage1>(p_in1(), p_in2(), p_tmp1()),
            make_independent(
                make_stage<stage1>(p_in1(), p_tmp1(), p_tmp2()), make_stage<stage2>(p_in2(), p_tmp1(), p_tmp3())),
            make_stage<stage2>(p_tmp2(), p_tmp3(), p_out()));

        // after last stage:
        //   p_out:  {0, 0, 0, 0, 0, 0}
        //   p_tmp3: {-1, 1, -1, 0, -1, 1}
        //   p_tmp2: {-1, 0, 0, 1, -1, 0}
        //
        // after second independent stage:
        //   p_out:  {0, 0, 0, 0, 0, 0}
        //   p_tmp3: {-1, 1, -1, 0, -1, 1}
        //   p_tmp2: {-1, 0, 0, 1, -1, 0}
        //   p_tmp1: {-2, 2, -2, 0, -2, 2}
        //   p_in2:  {-2, 1, -1, 1, -2, 1}
        //
        // after first independent stage:
        //   p_out:  {0, 0, 0, 0, 0, 0}
        //   p_tmp3: {-1, 1, -1, 0, -1, 1}
        //   p_tmp2: {-1, 0, 0, 1, -1, 0}
        //   p_tmp1: {-2, 2, -2, 1, -2, 2}
        //   p_in2:  {-2, 1, -1, 1, -2, 1}
        //   p_in1:  {-1, 1, -1, 1, -1, 1}
        //
        // after first stage
        //   p_out:  {0, 0, 0, 0, 0, 0}
        //   p_tmp3: {-1, 1, -1, 0, -1, 1}
        //   p_tmp2: {-1, 0, 0, 1, -1, 0}
        //   p_tmp1: {-2, 2, -2, 1, -2, 2}
        //   p_in2:  {-2, 3, -3, 1, -3, 3}
        //   p_in1:  {-2, 3, -3, 1, -2, 3}
        computation<p_in1, p_in2, p_tmp1, p_tmp2, p_tmp3, p_out> comp = make_computation<backend_t>(grid, mss_);

        EXPECT_EQ((rt_extent{-2, 3, -3, 1, -2, 3}), comp.get_arg_extent(p_in1()));
        EXPECT_EQ((rt_extent{-2, 3, -3, 1, -3, 3}), comp.get_arg_extent(p_in2()));
        EXPECT_EQ((rt_extent{-2, 2, -2, 1, -2, 2}), comp.get_arg_extent(p_tmp1()));
        EXPECT_EQ((rt_extent{-1, 0, 0, 1, -1, 0}), comp.get_arg_extent(p_tmp2()));
        EXPECT_EQ((rt_extent{-1, 1, -1, 0, -1, 1}), comp.get_arg_extent(p_tmp3()));
        EXPECT_EQ((rt_extent{0, 0, 0, 0, 0, 0}), comp.get_arg_extent(p_out()));

        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(intent::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_tmp1()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_tmp2()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_tmp3()));
        EXPECT_EQ(intent::inout, comp.get_arg_intent(p_out()));
    }
}
