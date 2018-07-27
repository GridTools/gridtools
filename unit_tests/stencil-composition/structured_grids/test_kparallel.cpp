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

#include "gtest/gtest.h"
#include <gridtools/stencil-composition/stencil-composition.hpp>

#include <gridtools/gridtools.hpp>

#include <boost/fusion/include/make_vector.hpp>

#include "backend_select.hpp"

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;
using gridtools::uint_t;

using namespace gridtools::enumtype;

namespace {

    template <typename Axis>
    struct parallel_functor {
        typedef accessor<0> in;
        typedef accessor<1, inout> out;
        typedef boost::mpl::vector<in, out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval, typename Axis::template get_interval<0>) {
            eval(out()) = eval(in());
        }
        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval, typename Axis::template get_interval<1>) {
            eval(out()) = 2 * eval(in());
        }
    };
} // namespace

TEST(structured_grid, kparallel) {

    constexpr uint_t d1 = 7;
    constexpr uint_t d2 = 8;
    constexpr uint_t d3_l = 14;
    constexpr uint_t d3_u = 16;

    using axis = gridtools::axis<2>;
    using storage_info_t = typename backend_t::storage_traits_t::storage_info_t<1, 3, gridtools::halo<0, 0, 0>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<double, storage_info_t>;

    storage_info_t storage_info(d1, d2, d3_l + d3_u);

    storage_t in(storage_info, [](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); });
    storage_t out(storage_info, (double)1.5);

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto grid = gridtools::make_grid(d1, d2, axis(d3_l, d3_u));

    auto copy = gridtools::make_computation<backend_t>(grid,
        p_in() = in,
        p_out() = out,
        gridtools::make_multistage(
            execute<parallel, 20>(), gridtools::make_stage<parallel_functor<axis>>(p_in(), p_out())));

    copy.run();

    copy.sync_bound_data_stores();

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

TEST(structured_grid, kparallel_with_extentoffsets_around_interval) {

    constexpr uint_t d1 = 7;
    constexpr uint_t d2 = 8;
    constexpr uint_t d3_l = 14;
    constexpr uint_t d3_u = 16;

    using axis = gridtools::axis<2>::with_offset_limit<5>::with_extra_offsets<3>;
    using storage_info_t = typename backend_t::storage_traits_t::storage_info_t<1, 3, gridtools::halo<0, 0, 0>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<double, storage_info_t>;

    storage_info_t storage_info(d1, d2, d3_l + d3_u);

    storage_t in(storage_info, [](int i, int j, int k) { return (double)(i * 1000 + j * 100 + k); });
    storage_t out(storage_info, (double)1.5);

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto grid = gridtools::make_grid(d1, d2, axis(d3_l, d3_u));

    auto copy = gridtools::make_computation<backend_t>(grid,
        p_in() = in,
        p_out() = out,
        gridtools::make_multistage(
            execute<parallel, 20>(), gridtools::make_stage<parallel_functor<axis>>(p_in(), p_out())));

    copy.run();

    copy.sync_bound_data_stores();

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
