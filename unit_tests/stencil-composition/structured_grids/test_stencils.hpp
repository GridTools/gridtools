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

#pragma once

#include <boost/fusion/include/make_vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

/*
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;

using namespace gridtools;
using namespace execute;

namespace copy_stencils_3D_2D_1D_0D {
    struct copy_functor {
        static const int n_args = 2;
        typedef accessor<0> in;
        typedef accessor<1, intent::inout> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    void handle_error(int) { std::cout << "error" << std::endl; }

    template <typename SrcLayout, typename DstLayout, typename T>
    bool test(int x, int y, int z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        using meta_dst_t = typename backend_t::storage_traits_t::
            select_custom_layout_storage_info<0, DstLayout, zero_halo<DstLayout::masked_length>>::type;
        using meta_src_t = typename backend_t::storage_traits_t::
            select_custom_layout_storage_info<0, SrcLayout, zero_halo<SrcLayout::masked_length>>::type;

        typedef backend_t::storage_traits_t::data_store_t<T, meta_dst_t> storage_t;
        typedef backend_t::storage_traits_t::data_store_t<T, meta_src_t> src_storage_t;

        meta_dst_t meta_dst_(d1, d2, d3);
        meta_src_t meta_src_(d1, d2, d3);

        std::function<T(int, int, int)> init_lambda = [](int i, int j, int k) { return (T)(i + j + k); };
        src_storage_t in(meta_src_, init_lambda);
        storage_t out(meta_dst_, (T)1.5);

        typedef arg<0, src_storage_t> p_in;
        typedef arg<1, storage_t> p_out;

        auto grid_ = make_grid(d1, d2, d3);

        auto copy = gridtools::make_computation<backend_t>(grid_,
            p_in() = in,
            p_out() = out,
            gridtools::make_multistage(execute::forward(), gridtools::make_stage<copy_functor>(p_in(), p_out())));

        copy.run();

        copy.sync_bound_data_stores();

        bool ok = true;
        auto outv = make_host_view(out);
        auto inv = make_host_view(in);
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                for (int k = 0; k < d3; ++k) {
                    if (inv(i, j, k) != outv(i, j, k)) {
                        ok = false;
                    }
                }

        return ok;
    }

} // namespace copy_stencils_3D_2D_1D_0D

constexpr int_t size0 = 59;
constexpr int_t size1 = 47;
constexpr int_t size2 = 71;

TEST(TESTCLASS, copies3D) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, 1, 2>, gridtools::layout_map<0, 1, 2>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies3Dtranspose) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2, 1, 0>, gridtools::layout_map<0, 1, 2>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Dij) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, 1, -1>, gridtools::layout_map<0, 1, 2>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Dik) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, -1, 1>, gridtools::layout_map<0, 1, 2>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Djk) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, 0, 1>, gridtools::layout_map<0, 1, 2>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Di) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, -1, -1>, gridtools::layout_map<0, 1, 2>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Dj) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, 0, -1>, gridtools::layout_map<0, 1, 2>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2Dk) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, -1, 0>, gridtools::layout_map<0, 1, 2>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DScalar) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, -1, -1>, gridtools::layout_map<0, 1, 2>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies3DDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, 1, 2>, gridtools::layout_map<2, 0, 1>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies3DtransposeDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<2, 1, 0>, gridtools::layout_map<2, 0, 1>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DijDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1, 0, -1>, gridtools::layout_map<2, 0, 1>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DikDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<1, -1, 0>, gridtools::layout_map<2, 0, 1>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DjkDst) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, 1, 0>, gridtools::layout_map<2, 0, 1>, double>(
                  size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DiDst) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, -1, -1>, gridtools::layout_map<2, 0, 1>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DjDst) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, 0, -1>, gridtools::layout_map<2, 0, 1>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DkDst) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, -1, 0>, gridtools::layout_map<2, 0, 1>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies2DScalarDst) {
    EXPECT_EQ(
        (copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<-1, -1, -1>, gridtools::layout_map<2, 0, 1>, double>(
            size0, size1, size2)),
        true);
}

TEST(TESTCLASS, copies3D_bool) {
    EXPECT_EQ((copy_stencils_3D_2D_1D_0D::test<gridtools::layout_map<0, 1, 2>, gridtools::layout_map<0, 1, 2>, bool>(
                  size0, size1, size2)),
        true);
}
