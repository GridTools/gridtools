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

#include <gridtools/stencil-composition/level.hpp>
#include <gridtools/stencil-composition/level_metafunctions.hpp>

using namespace gridtools;

TEST(test_level, leq) {
    using lower_level = level< 0, -1 >;
    using greater_level = level< 1, 1 >;

    ASSERT_TRUE((level_leq< lower_level, greater_level >::value));
    ASSERT_FALSE((level_leq< greater_level, lower_level >::value));
}

TEST(test_level, leq_same_splitter) {
    using lower_level = level< 1, -1 >;
    using greater_level = level< 1, 1 >;

    ASSERT_TRUE((level_leq< lower_level, greater_level >::value));
    ASSERT_FALSE((level_leq< greater_level, lower_level >::value));
}

TEST(test_level, leq_equal_levels) {
    using level1 = level< 1, -1 >;
    using level2 = level1;

    ASSERT_TRUE((level_leq< level1, level2 >::value));
    ASSERT_TRUE((level_leq< level2, level1 >::value));
}

TEST(test_level, lt) {
    using lower_level = level< 0, -1 >;
    using greater_level = level< 1, 1 >;

    ASSERT_TRUE((level_lt< lower_level, greater_level >::value));
    ASSERT_FALSE((level_lt< greater_level, lower_level >::value));
}

TEST(test_level, lt_equal_levels) {
    using level1 = level< 1, -1 >;
    using level2 = level1;

    ASSERT_FALSE((level_lt< level1, level2 >::value));
    ASSERT_FALSE((level_lt< level2, level1 >::value));
}

TEST(test_level, geq) {
    using lower_level = level< 0, -1 >;
    using greater_level = level< 1, 1 >;

    ASSERT_TRUE((level_geq< greater_level, lower_level >::value));
    ASSERT_FALSE((level_geq< lower_level, greater_level >::value));
}
