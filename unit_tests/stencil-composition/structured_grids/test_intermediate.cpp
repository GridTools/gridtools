/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <iostream>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace test_intermediate {
    using namespace gridtools;
    using namespace enumtype;

    struct stage1 {
        using in1 = accessor<0, enumtype::in, extent<0, 1, -1, 0, 0, 1>>;
        using in2 = accessor<1, enumtype::in, extent<0, 1, -1, 0, -1, 1>>;
        using out = accessor<2, enumtype::inout, extent<>>;
        using arg_list = make_arg_list<in1, in2, out>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };

    struct stage2 {
        using in1 = accessor<0, enumtype::in, extent<-1, 0, 0, 1, -1, 0>>;
        using in2 = accessor<1, enumtype::in, extent<-1, 1, -1, 0, -1, 1>>;
        using out = accessor<2, enumtype::inout, extent<>>;
        using arg_list = make_arg_list<in1, in2, out>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };
} // namespace test_intermediate

namespace {
    std::ostream &operator<<(std::ostream &os, const gridtools::rt_extent &extent) {
        return (os << extent.iminus << ":" << extent.iplus << ", " << extent.jminus << ":" << extent.jplus << ", "
                   << extent.kminus << ":" << extent.kplus);
    }
} // namespace

TEST(intermediate, test_get_arg_functions) {
    using namespace test_intermediate;

    using storage_info_t = storage_traits<backend_t::backend_id_t>::storage_info_t<0, 1>;
    using data_store_t = storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t>;

    using p_in1 = arg<0, data_store_t>;
    using p_in2 = arg<1, data_store_t>;
    using p_tmp1 = arg<2, data_store_t>;
    using p_tmp2 = arg<3, data_store_t>;
    using p_tmp3 = arg<4, data_store_t>;
    using p_out = arg<5, data_store_t>;

    halo_descriptor di = {0, 0, 0, 2, 5};
    halo_descriptor dj = {0, 0, 0, 2, 5};

    auto grid = gridtools::make_grid(di, dj, 3);
    {
        auto mss_ =
            make_multistage(enumtype::execute<enumtype::forward>(), make_stage<stage1>(p_in1(), p_in2(), p_out()));
        computation<p_in1, p_in2, p_out> comp = make_computation<backend_t>(grid, mss_);

        EXPECT_EQ((rt_extent{0, 1, -1, 0, 0, 1}), comp.get_arg_extent(p_in1()));
        EXPECT_EQ((rt_extent{0, 1, -1, 0, -1, 1}), comp.get_arg_extent(p_in2()));
        EXPECT_EQ((rt_extent{0, 0, 0, 0, 0, 0}), comp.get_arg_extent(p_out()));

        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_out()));
    }

    {
        auto mss_ = make_multistage(enumtype::execute<enumtype::forward>(),
            make_stage<stage1>(p_in1(), p_in2(), p_tmp1()),
            make_stage<stage2>(p_in1(), p_tmp1(), p_out()));
        computation<p_in1, p_in2, p_tmp1, p_out> comp = make_computation<backend_t>(grid, mss_);

        EXPECT_EQ((rt_extent{-1, 2, -2, 1, -1, 2}), comp.get_arg_extent(p_in1()));
        EXPECT_EQ((rt_extent{-1, 2, -2, 0, -2, 2}), comp.get_arg_extent(p_in2()));
        EXPECT_EQ((rt_extent{-1, 1, -1, 0, -1, 1}), comp.get_arg_extent(p_tmp1()));
        EXPECT_EQ((rt_extent{0, 0, 0, 0, 0, 0}), comp.get_arg_extent(p_out()));

        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_tmp1()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_out()));
    }

    {
        auto mss_ = make_multistage(enumtype::execute<enumtype::forward>(),
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

        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_tmp1()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_tmp2()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_tmp3()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_out()));
    }
    {
        auto mss_ = make_multistage(enumtype::execute<enumtype::forward>(),
            make_stage_with_extent<stage1, extent<>>(p_in1(), p_in2(), p_out()));
        computation<p_in1, p_in2, p_out> comp = make_computation<backend_t>(grid, mss_);

#ifndef __CUDACC__
        EXPECT_ANY_THROW(comp.get_arg_extent(p_in1()));
#endif

        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in1()));
        EXPECT_EQ(enumtype::in, comp.get_arg_intent(p_in2()));
        EXPECT_EQ(enumtype::inout, comp.get_arg_intent(p_out()));
    }
}
