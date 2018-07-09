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

#include "../test_helper.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil-composition/interval.hpp>

using namespace gridtools;

TEST(test_interval, modify) {
    using my_interval = interval<level<0, -1>, level<1, -1>>;

    ASSERT_TYPE_EQ<interval<level<0, -2>, level<1, -1>>, my_interval::modify<-1, 0>>();
    ASSERT_TYPE_EQ<interval<level<0, 1>, level<1, 1>>, my_interval::modify<1, 1>>();
    ASSERT_TYPE_EQ<interval<level<0, -3>, level<1, -1>>, my_interval::modify<-2, 0>>();
    ASSERT_TYPE_EQ<interval<level<0, 2>, level<1, 2>>, my_interval::modify<2, 2>>();
}

TEST(test_interval, join) {
    using interval1 = interval<level<1, -2>, level<1, -1>>;
    using interval2 = interval<level<0, -1>, level<3, -1>>;
    using interval3 = interval<level<2, -2>, level<3, -1>>;
    using joined_interval = join_interval<interval1, interval2, interval3>;

    ASSERT_TYPE_EQ<interval<level<0, -1>, level<3, -1>>, joined_interval>();
}
