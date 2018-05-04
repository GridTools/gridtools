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

#include <cmath>

#include "gtest/gtest.h"
#include <common/numerics.hpp>

using namespace gridtools;

TEST(numerics, pow3) {
    constexpr int x0 = _impl::static_pow3< 0 >::value;
    constexpr int x1 = _impl::static_pow3< 1 >::value;
    constexpr int x2 = _impl::static_pow3< 2 >::value;
    constexpr int x3 = _impl::static_pow3< 3 >::value;
    constexpr int x4 = _impl::static_pow3< 4 >::value;
    EXPECT_EQ(x0, 1);
    EXPECT_EQ(x1, 3);
    EXPECT_EQ(x2, 9);
    EXPECT_EQ(x3, 27);
    EXPECT_EQ(x4, 81);
}

TEST(numerics, static_ceil) {
    constexpr float x = 3.1415;
    constexpr auto r1 = _impl::static_ceil(x / 1.0);
    constexpr auto r2 = _impl::static_ceil(x / -1.0);
    constexpr auto r3 = _impl::static_ceil(x / -2.0);
    constexpr auto r4 = _impl::static_ceil(x / 2.0);
    EXPECT_EQ(r1, std::ceil(x / 1.0));
    EXPECT_EQ(r2, std::ceil(x / -1.0));
    EXPECT_EQ(r3, std::ceil(x / -2.0));
    EXPECT_EQ(r4, std::ceil(x / 2.0));
}
