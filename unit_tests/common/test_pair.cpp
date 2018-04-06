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

#include "common/pair.hpp"
#include "gtest/gtest.h"
#include <type_traits>

TEST(pair, non_uniform_ctor) {
    int int_val = 1;
    size_t size_t_val = 2;

    gridtools::pair< size_t, size_t > my_pair(int_val, size_t_val);

    EXPECT_EQ((size_t)int_val, my_pair.first);
    EXPECT_EQ(size_t_val, my_pair.second);
}

TEST(pair, get_rval_ref) {
    size_t val0 = 1;
    size_t val1 = 2;

    EXPECT_TRUE(std::is_rvalue_reference< decltype(
            gridtools::get< 0 >(gridtools::pair< size_t, size_t >{val0, val1})) >::value);
    EXPECT_EQ(val0, gridtools::get< 0 >(gridtools::pair< size_t, size_t >{val0, val1}));
    EXPECT_EQ(val1, gridtools::get< 1 >(gridtools::pair< size_t, size_t >{val0, val1}));
}

TEST(pair, eq) {
    gridtools::pair< size_t, size_t > pair1{1, 2};
    gridtools::pair< size_t, size_t > pair2{pair1};

    EXPECT_TRUE(pair1 == pair2);
    EXPECT_FALSE(pair1 != pair2);
    EXPECT_FALSE(pair1 < pair2);
    EXPECT_TRUE(pair1 <= pair2);
    EXPECT_FALSE(pair1 > pair2);
    EXPECT_TRUE(pair1 >= pair2);
}
TEST(pair, compare_first_differ) {
    gridtools::pair< size_t, size_t > smaller{1, 2};
    gridtools::pair< size_t, size_t > bigger{2, 2};

    EXPECT_FALSE(smaller == bigger);
    EXPECT_TRUE(smaller != bigger);
    EXPECT_TRUE(smaller < bigger);
    EXPECT_TRUE(smaller <= bigger);
    EXPECT_FALSE(smaller > bigger);
    EXPECT_FALSE(smaller >= bigger);
}

TEST(pair, lt_gt_second_differ) {
    gridtools::pair< size_t, size_t > smaller{1, 1};
    gridtools::pair< size_t, size_t > bigger{1, 2};

    EXPECT_FALSE(smaller == bigger);
    EXPECT_TRUE(smaller != bigger);
    EXPECT_TRUE(smaller < bigger);
    EXPECT_TRUE(smaller <= bigger);
    EXPECT_FALSE(smaller > bigger);
    EXPECT_FALSE(smaller >= bigger);
}

TEST(pair, construct_from_std_pair) {
    std::pair< size_t, size_t > std_pair{1, 2};

    gridtools::pair< size_t, size_t > gt_pair(std_pair);

    ASSERT_EQ(std_pair.first, gt_pair.first);
    ASSERT_EQ(std_pair.second, gt_pair.second);
}
