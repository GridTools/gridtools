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
#include <common/generic_metafunctions/is_pack_of.hpp>

using namespace gridtools;

template < typename T >
struct is_int : boost::mpl::false_ {};
template <>
struct is_int< int > : boost::mpl::true_ {};

template < typename... Int, typename = is_pack_of< is_int, Int... > >
GT_FUNCTION constexpr int test_fn(Int...) {
    return 1;
}

GT_FUNCTION
constexpr int test_fn(double, double) { return 2; }

TEST(is_offset_of, int) { GRIDTOOLS_STATIC_ASSERT((test_fn(int(3), int(4)) == 1), "ERROR"); }

TEST(is_offset_of, empty) { GRIDTOOLS_STATIC_ASSERT((test_fn() == 1), "ERROR"); }

TEST(is_offset_of, long) { GRIDTOOLS_STATIC_ASSERT((test_fn(long(3), int(4)) == 2), "ERROR"); }

template < typename... Ts, typename = is_pack_of_with_placeholder< std::is_same< int, boost::mpl::_ >, Ts... > >
GT_FUNCTION constexpr int test_is_same_as_int(Ts...) {
    return 1;
}

GT_FUNCTION
constexpr int test_is_same_as_int(double, double) { return 2; }

TEST(is_pack_of_with_placeholder, int_is_same_as_int) {
    GRIDTOOLS_STATIC_ASSERT((test_is_same_as_int(int(), int()) == 1), "ERROR");
}
TEST(is_pack_of_with_placeholder, unsigned_int_is_not_same_as_int) {
    GRIDTOOLS_STATIC_ASSERT((test_is_same_as_int(uint_t(), int()) == 2), "ERROR");
}
TEST(is_pack_of_with_placeholder, empty_is_accepted) { GRIDTOOLS_STATIC_ASSERT((test_is_same_as_int() == 1), "ERROR"); }
