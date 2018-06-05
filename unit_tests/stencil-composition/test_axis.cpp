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
#include "stencil-composition/axis.hpp"
#include "../test_helper.hpp"

using namespace gridtools;

TEST(test_axis, ctor) {
    auto axis_ = axis< 2 >((uint_t)5, (uint_t)4);

    ASSERT_EQ(5, axis_.interval_size(0));
    ASSERT_EQ(4, axis_.interval_size(1));
}

TEST(test_axis, intervals) {
    using axis_t = axis< 3 >;

    // full interval
    ASSERT_TYPE_EQ< interval< level< 0, 1 >, level< 3, -1 > >, axis_t::full_interval >();

    // intervals by id
    ASSERT_TYPE_EQ< interval< level< 0, 1 >, level< 1, -1 > >, axis_t::get_interval< 0 > >();
    ASSERT_TYPE_EQ< interval< level< 1, 1 >, level< 2, -1 > >, axis_t::get_interval< 1 > >();

    // hull of multiple intervals
    ASSERT_TYPE_EQ< interval< level< 1, 1 >, level< 3, -1 > >, axis_t::get_interval< 1, 2 > >();
}
