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
#include <common/defs.hpp>
#include <common/generic_metafunctions/all_integrals.hpp>

using namespace gridtools;

TEST(AllIntegrals, is_all_integral) {
    GRIDTOOLS_STATIC_ASSERT((is_all_integral< int, u_int, long >::value), "not all integral");
    GRIDTOOLS_STATIC_ASSERT((!is_all_integral< int, double >::value), "all integral (but shouldn't be)");
}

TEST(AllIntegrals, is_all_static_integral) {
    GRIDTOOLS_STATIC_ASSERT((is_all_static_integral< static_int< 0 >,
                                static_uint< 1 >,
                                static_short< 2 >,
                                static_ushort< 3 >,
                                static_bool< 1 > >::value),
        "not all static integral");
}

template < typename... Ts, typename = all_integral< Ts... > >
bool is_enabled_function(Ts... vals) {
    return true;
}

TEST(AllIntegrals, enable_with_all_integral) { ASSERT_TRUE(is_enabled_function(1, 2, 3)); }
TEST(AllIntegrals, enable_with_all_integral_empty) { ASSERT_TRUE(is_enabled_function()); }
