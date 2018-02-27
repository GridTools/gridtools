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
#include <stencil-composition/stencil-composition.hpp>

#include "backend_select.hpp"

namespace test_conditionals {
    using namespace gridtools;

    using axis_t = axis< 1 >;
    using x_interval = axis_t::full_interval;

    template < uint_t Id >
    struct functor {

        typedef accessor< 0, enumtype::inout > p_dummy;
        typedef boost::mpl::vector1< p_dummy > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(p_dummy()) = +Id;
        }
    };

    bool test() {

        auto cond = []() { return false; };
        auto cond2 = []() { return true; };

        auto grid_ = make_grid((uint_t)2, (uint_t)2, axis_t((uint_t)3));

        typedef gridtools::storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
        typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, storage_info_t >
            data_store_t;
        storage_info_t meta_data_(3, 3, 3);
        data_store_t dummy(meta_data_, 0.);
        typedef arg< 0, data_store_t > p_dummy;

        typedef boost::mpl::vector1< p_dummy > arg_list;
        aggregator_type< arg_list > domain_(dummy);

        auto comp_ = make_computation< backend_t >(
            domain_,
            grid_,
            if_(cond,
                make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor< 0 > >(p_dummy())),
                if_(cond2,
                    make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor< 1 > >(p_dummy())),
                    make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor< 2 > >(p_dummy())))));

        bool result = true;
        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();
        result = result && (make_host_view(dummy)(0, 0, 0) == 1);
        return result;
    }
} // namespace test_conditional

TEST(stencil_composition, conditionals) { EXPECT_TRUE(test_conditionals::test()); }
