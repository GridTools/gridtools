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
#include "common/defs.hpp"
#include "common/gt_math.hpp"

using namespace gridtools;

TEST(math, test_min) {
    ASSERT_TRUE(math::min(5, 2, 7) == 2);
    ASSERT_TRUE(math::min(5, -1) == -1);

    ASSERT_REAL_EQ(math::min(5.3, 22.0, 7.7), 5.3);

    // checking returned by const &
    double a = 3.5;
    double b = 2.3;
    double const &min = math::min(a, b);
    ASSERT_REAL_EQ(min, 2.3);
    b = 8;
    ASSERT_REAL_EQ(min, 8);
}

TEST(math, test_max) {
    ASSERT_TRUE(math::max(5, 2, 7) == 7);
    ASSERT_TRUE(math::max(5, -1) == 5);

    ASSERT_REAL_EQ(math::max(5.3, 22.0, 7.7), 22.0);
    // checking returned by const &
    double a = 3.5;
    double b = 2.3;
    double const &max = math::max(a, b);

    ASSERT_REAL_EQ(max, 3.5);
    a = 8;
    ASSERT_REAL_EQ(max, 8);
}

TEST(math, test_fabs) {
    ASSERT_REAL_EQ(math::fabs(5.6), 5.6);
    ASSERT_REAL_EQ(math::fabs(-5.6), 5.6);
}

TEST(math, test_abs) {
    ASSERT_TRUE(math::abs(5.6) == 5);
    ASSERT_TRUE(math::abs(-5.6) == 5);
}
