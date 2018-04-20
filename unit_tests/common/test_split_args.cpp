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

#include <common/split_args.hpp>

#include <tuple>
#include <type_traits>
#include <gtest/gtest.h>

namespace gridtools {

    template < class Testee, class FirstSink, class SecondSink >
    void helper(Testee &&testee, FirstSink first_sink, SecondSink second_sink) {
        first_sink(std::forward< Testee >(testee).first);
        second_sink(std::forward< Testee >(testee).second);
    }

    TEST(raw_split_args, functional) {
        int val = 1;
        const int c_val = 2;
        helper(raw_split_args< std::is_lvalue_reference >(42, c_val, 0., val, c_val),
            [](std::tuple< int const &, int &, int const & > const &x) { EXPECT_EQ(std::make_tuple(2, 1, 2), x); },
            [](std::tuple< int &&, double && > const &x) { EXPECT_EQ(std::make_tuple(42, 0.), x); });
    }

    TEST(split_args, functional) {
        int ival = 1;
        const double dval = 2;
        helper(split_args< std::is_integral >(42, dval, 0., ival),
            [](std::tuple< int &&, int & > const &x) { EXPECT_EQ(std::make_tuple(42, 1), x); },
            [](std::tuple< const double &, double && > const &x) { EXPECT_EQ(std::make_tuple(2., 0.), x); });
    }
}
