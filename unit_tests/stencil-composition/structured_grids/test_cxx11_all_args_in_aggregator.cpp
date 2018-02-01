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

#include <common/gt_assert.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

namespace all_args_in_aggregator {
    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;

    struct copy_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    TEST(testdomain, testindices) {
        typedef storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
        typedef storage_traits< backend_t::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;
        typedef arg< 0, data_store_t > p_in;
        typedef tmp_arg< 2, data_store_t > p_tmp;
        typedef arg< 1, data_store_t > p_out;
        typedef arg< 3, data_store_t > p_err;

        typedef boost::mpl::vector< p_in, p_out, p_tmp > accessor_list;
        using agg = aggregator_type< accessor_list >;

        auto bad = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage< copy_functor >(p_in(), p_err()),
            make_stage< copy_functor >(p_in(), p_out()));

        auto good = make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage< copy_functor >(p_in(), p_tmp()),
            make_stage< copy_functor >(p_tmp(), p_out()));

        GRIDTOOLS_STATIC_ASSERT((_impl::all_args_in_aggregator< agg, decltype(good) >::type::value), "");
        GRIDTOOLS_STATIC_ASSERT((!_impl::all_args_in_aggregator< agg, decltype(bad) >::type::value), "");

        GRIDTOOLS_STATIC_ASSERT(
            (_impl::all_args_in_aggregator< agg, decltype(good), decltype(good) >::type::value), "");
        GRIDTOOLS_STATIC_ASSERT(
            (!_impl::all_args_in_aggregator< agg, decltype(bad), decltype(good) >::type::value), "");
    }

} // namespace all_args_in_aggregator
