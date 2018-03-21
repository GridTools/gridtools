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

#include <gtest/gtest.h>
#include "common/boost_pp_generic_macros.hpp"

TEST(double_parenthesis, check) {
#define my_types (int, 2)(double, 3)
#define my_types_double_parenthesis GRIDTOOLS_PP_SEQ_DOUBLE_PARENS(my_types)
    ASSERT_EQ(std::string("((int, 2)) ((double, 3))"), std::string(BOOST_PP_STRINGIZE(my_types_double_parenthesis)));
#undef my_types
#undef my_types_double_parenthesis
}

#define my_types ((int))((double))
GRIDTOOLS_PP_MAKE_VARIANT(myvariant, my_types);
#undef my_types
TEST(variant, automatic_conversion) {
    myvariant v = 3;
    int i = v;

    v = 3.;
    double d = v;

    try {
        int j = v;
        ASSERT_TRUE(false);
    } catch (const boost::bad_get &e) {
        ASSERT_TRUE(true);
    }
}

#define my_types ((int, 3))((double, 1))
GRIDTOOLS_PP_MAKE_VARIANT(myvariant_tuple, my_types);
#undef my_types
TEST(variant_with_tuple, automatic_conversion) {
    myvariant_tuple v = 3;
    int i = v;

    v = 3.;
    double d = v;

    try {
        int j = v;
        ASSERT_TRUE(false);
    } catch (const boost::bad_get &e) {
        ASSERT_TRUE(true);
    }
}
