/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "common/generic_metafunctions/gt_integer_sequence.hpp"
#include "common/array.hpp"
#include "common/generic_metafunctions/variadic_typedef.hpp"

template < int Idx >
struct get_component {

    GT_FUNCTION
    constexpr get_component() {}

    template < typename... Ints >
    GT_FUNCTION constexpr static int apply(Ints... args_) {
        return std::get< Idx >(std::make_tuple(args_...));
    }
};
using namespace gridtools;

TEST(integer_sequence, fill_array) {

    using seq = gridtools::apply_gt_integer_sequence< typename gridtools::make_gt_integer_sequence< int, 4 >::type >;

    // calling the array constexpr copy constructor
    constexpr gridtools::array< int, 4 > out(
        seq::template apply< gridtools::array< int, 4 >, get_component >(0, 1, 2, 3));

    // verifying that the information is actually compile-time known and that it's correct
    GRIDTOOLS_STATIC_ASSERT(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3, "Error in test_integer_sequence");
}

template < int Idx >
struct transform {

    GT_FUNCTION
    constexpr transform() {}

    template < typename... Args >
    GT_FUNCTION constexpr static int apply(Args... args) {
        return get_from_variadic_pack< Idx >::apply(args...) - Idx;
    }
};

struct lambda {
    constexpr int operator()(const int i, const int j, const int k, const int l, const int add) {
        return add * (i + j + k + l);
    }
};

TEST(integer_sequence, apply_lambda) {

    using seq = gridtools::apply_gt_integer_sequence< typename gridtools::make_gt_integer_sequence< int, 4 >::type >;

    constexpr auto gather = lambda();

    constexpr int result = seq::template apply_lambda< int, decltype(gather), transform >(gather, 17, 4, 6, 34, 5);

    GRIDTOOLS_STATIC_ASSERT((static_int< result >::value == 731), "ERROR");

    ASSERT_TRUE(true);
}

