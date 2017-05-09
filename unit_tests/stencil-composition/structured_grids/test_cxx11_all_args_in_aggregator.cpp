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
#include "stencil-composition/stencil-composition.hpp"

namespace all_args_in_aggregator {
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;

    struct copy_functor {

        typedef gridtools::accessor< 0, gridtools::enumtype::in, gridtools::extent<>, 3 > in;
        typedef gridtools::accessor< 1, gridtools::enumtype::inout, gridtools::extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct storage_type {
        using iterator = int;
        using value_type = float;
    };

    TEST(testdomain, testindices) {
        typedef gridtools::arg< 0, storage_type > p_in;
        typedef gridtools::arg< 1, storage_type > p_out;
        typedef gridtools::arg< 2, storage_type > p_err;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        using agg = gridtools::aggregator_type< accessor_list >;

        auto bad = gridtools::make_multistage(gridtools::enumtype::execute< gridtools::enumtype::forward >(),
            gridtools::make_stage< copy_functor >(p_in(), p_err()),
            gridtools::make_stage< copy_functor >(p_in(), p_out()));

        auto good = gridtools::make_multistage(gridtools::enumtype::execute< gridtools::enumtype::forward >(),
            gridtools::make_stage< copy_functor >(p_in(), p_out()),
            gridtools::make_stage< copy_functor >(p_in(), p_out()));

        static_assert(gridtools::_impl::all_args_in_aggregator< agg, decltype(good) >::type::value, "");
        static_assert(!gridtools::_impl::all_args_in_aggregator< agg, decltype(bad) >::type::value, "");

        static_assert(gridtools::_impl::all_args_in_aggregator< agg, decltype(good), decltype(good) >::type::value, "");
        static_assert(!gridtools::_impl::all_args_in_aggregator< agg, decltype(bad), decltype(good) >::type::value, "");
    }

} // namespace all_args_in_aggregator
