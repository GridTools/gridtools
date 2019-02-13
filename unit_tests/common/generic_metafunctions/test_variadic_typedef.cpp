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
#include <gridtools/common/generic_metafunctions/variadic_typedef.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(variadic_typedef, test) {

    typedef variadic_typedef<int, double, unsigned int> tt;

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<0>::type, int>::value), "Error");

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<1>::type, double>::value), "Error");

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<2>::type, unsigned int>::value), "Error");
}

TEST(variadic_typedef, get_from_variadic_pack) {

    GT_STATIC_ASSERT((static_int<get_from_variadic_pack<3>::apply(2, 6, 8, 3, 5)>::value == 3), "Error");

    GT_STATIC_ASSERT(
        (static_int<get_from_variadic_pack<7>::apply(2, 6, 8, 3, 5, 4, 6, -8, 4, 3, 1, 54, 67)>::value == -8), "Error");
}

TEST(variadic_typedef, find) {

    typedef variadic_typedef<int, double, unsigned int, double> tt;

    GT_STATIC_ASSERT((tt::find<int>() == 0), "ERROR");
    GT_STATIC_ASSERT((tt::find<double>() == 1), "ERROR");
    GT_STATIC_ASSERT((tt::find<unsigned int>() == 2), "ERROR");
}
