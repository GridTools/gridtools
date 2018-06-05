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
#define PEDANTIC_DISABLED // too stringent for this test

#include "common/defs.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/structured_grids/accessor.hpp"
#include "gtest/gtest.h"
#include <iostream>

namespace test_iterate_domain {
    using namespace gridtools;
    using namespace enumtype;

    // This is the definition of the special regions in the "vertical" direction
    struct stage1 {
        typedef accessor< 0, enumtype::in, extent< 42, 42, 42, 42 >, 6 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 4 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };

    struct stage2 {
        typedef accessor< 0, enumtype::in, extent< 42, 42, 42, 42 >, 6 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 4 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };
} // namespace test_iterate_domain

TEST(testdomain, iterate_domain_with_extents) {
    using namespace test_iterate_domain;
    typedef backend< enumtype::Host, enumtype::structured, enumtype::Naive > backend_t;

    typedef storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< backend_t::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;

    halo_descriptor di = {0, 0, 0, 2, 5};
    halo_descriptor dj = {0, 0, 0, 2, 5};

    auto grid = gridtools::make_grid(di, dj, 3);
    {
        auto mss_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage_with_extent< stage1, extent< 0, 1, 0, 0 > >(p_in(), p_out()));
        auto computation_ = make_computation< backend< Host, GRIDBACKEND, Naive > >(grid, mss_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same< intermediate_t::extent_map_t, boost::mpl::void_ >::value, "extent computation happened");
    }
    {
        auto mss_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage_with_extent< stage1, extent< 0, 1, 0, 0 > >(p_in(), p_out()),
            make_stage_with_extent< stage2, extent< 0, 1, -1, 2 > >(p_out(), p_in()));
        auto computation_ = make_computation< backend< Host, GRIDBACKEND, Naive > >(grid, mss_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same< intermediate_t::extent_map_t, boost::mpl::void_ >::value, "extent computation happened");
    }
    {
        auto mss1_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage_with_extent< stage1, extent< 0, 1, 0, 0 > >(p_in(), p_out()),
            make_stage_with_extent< stage2, extent< 0, 1, -1, 2 > >(p_out(), p_in()));

        auto mss2_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage_with_extent< stage1, extent< -2, 1, 0, 0 > >(p_in(), p_out()),
            make_stage_with_extent< stage2, extent< -2, 1, -1, 2 > >(p_out(), p_in()));

        auto computation_ = make_computation< backend< Host, GRIDBACKEND, Naive > >(grid, mss1_, mss2_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same< intermediate_t::extent_map_t, boost::mpl::void_ >::value, "extent computation happened");
    }
    {
        auto mss1_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_independent(make_stage_with_extent< stage1, extent< 0, 1, 0, 0 > >(p_in(), p_out()),
                                         make_stage_with_extent< stage2, extent< 0, 1, -1, 2 > >(p_out(), p_in())));

        auto mss2_ = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage_with_extent< stage1, extent< -2, 1, 0, 0 > >(p_in(), p_out()),
            make_stage_with_extent< stage2, extent< -2, 1, -1, 2 > >(p_out(), p_in()));

        auto computation_ = make_computation< backend< Host, GRIDBACKEND, Naive > >(grid, mss1_, mss2_);

        typedef decltype(computation_) intermediate_t;
        static_assert(
            std::is_same< intermediate_t::extent_map_t, boost::mpl::void_ >::value, "extent computation happened");
    }
}
