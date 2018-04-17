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
#include <common/generic_metafunctions/gt_remove_qualifiers.hpp>
#include <stencil-composition/stencil-functions/stencil-functions.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <test_helper.hpp>
#include "backend_select.hpp"
#include "gtest/gtest.h"

/**
 * Compile-time test to ensure that types are correct in all call_proc stages
 */

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

namespace {
    // used to ensure that types are correctly passed between function calls (no implicit conversion)
    template < typename tag >
    struct special_type {};

    struct in_tag {};
    struct out_tag {};
}

class call_proc_stress_types : public testing::Test {
  protected:
    using storage_info_t = gridtools::storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 >;
    using data_store_in_t =
        gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< special_type< in_tag >, storage_info_t >;
    using data_store_out_t =
        gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< special_type< out_tag >, storage_info_t >;

    gridtools::grid< gridtools::axis< 1 >::axis_interval_t > grid;

    data_store_in_t in;
    data_store_out_t out;

    typedef arg< 0, data_store_in_t > p_in;
    typedef arg< 1, data_store_out_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain;

    call_proc_stress_types()
        : grid(make_grid(1, 1, 1)), in(storage_info_t{1, 1, 1}), out(storage_info_t{1, 1, 1}), domain(in, out) {}
};

namespace {
    struct local_tag {};

    struct triple_nesting_with_type_switching_third_stage {
        typedef inout_accessor< 0 > out;
        typedef in_accessor< 1 > local;
        typedef boost::mpl::vector< out, local > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            using out_type = typename remove_qualifiers< decltype(eval(out{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< out_tag >, out_type >{};

            using local_type = typename remove_qualifiers< decltype(eval(local{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< local_tag >, local_type >{};
        }
    };

    struct triple_nesting_with_type_switching_second_stage {
        typedef in_accessor< 0 > in;
        typedef inout_accessor< 1 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            using out_type = typename remove_qualifiers< decltype(eval(out{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< out_tag >, out_type >{};

            using in_type = typename remove_qualifiers< decltype(eval(in{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< in_tag >, in_type >{};

            special_type< local_tag > local{};

            call_proc< triple_nesting_with_type_switching_third_stage >::with(eval, out(), local);
        }
    };

    struct triple_nesting_with_type_switching_first_stage {
        typedef inout_accessor< 0 > out;
        typedef in_accessor< 1 > in;
        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            using out_type = typename remove_qualifiers< decltype(eval(out{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< out_tag >, out_type >{};

            using in_type = typename remove_qualifiers< decltype(eval(in{})) >::type;
            (void)ASSERT_TYPE_EQ< special_type< in_tag >, in_type >{};

            call_proc< triple_nesting_with_type_switching_second_stage >::with(eval, in(), out());
        }
    };
}

TEST_F(call_proc_stress_types, triple_nesting_with_type_switching) {
    auto comp = gridtools::make_computation< backend_t >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< triple_nesting_with_type_switching_first_stage >(p_out(), p_in())));
}
