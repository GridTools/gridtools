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

#include <gridtools/stencil-composition/level.hpp>
#include <gridtools/stencil-composition/level_metafunctions.hpp>

using namespace gridtools;

TEST(test_level, level_to_index) {
    ASSERT_TYPE_EQ<typename level_to_index<level<0, -2, 2>>::type, level_index<0, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, -1, 2>>::type, level_index<1, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, 1, 2>>::type, level_index<2, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<0, 2, 2>>::type, level_index<3, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, -2, 2>>::type, level_index<4, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, -1, 2>>::type, level_index<5, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, 1, 2>>::type, level_index<6, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<1, 2, 2>>::type, level_index<7, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, -2, 2>>::type, level_index<8, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, -1, 2>>::type, level_index<9, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, 1, 2>>::type, level_index<10, 2>>();
    ASSERT_TYPE_EQ<typename level_to_index<level<2, 2, 2>>::type, level_index<11, 2>>();
}

TEST(test_level, index_to_level) {
    ASSERT_TYPE_EQ<level<0, -2, 2>, typename index_to_level<level_index<0, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, -1, 2>, typename index_to_level<level_index<1, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, 1, 2>, typename index_to_level<level_index<2, 2>>::type>();
    ASSERT_TYPE_EQ<level<0, 2, 2>, typename index_to_level<level_index<3, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, -2, 2>, typename index_to_level<level_index<4, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, -1, 2>, typename index_to_level<level_index<5, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, 1, 2>, typename index_to_level<level_index<6, 2>>::type>();
    ASSERT_TYPE_EQ<level<1, 2, 2>, typename index_to_level<level_index<7, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, -2, 2>, typename index_to_level<level_index<8, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, -1, 2>, typename index_to_level<level_index<9, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, 1, 2>, typename index_to_level<level_index<10, 2>>::type>();
    ASSERT_TYPE_EQ<level<2, 2, 2>, typename index_to_level<level_index<11, 2>>::type>();
}

TEST(test_level, level_gt) {
    using level_1 = level<0, -1, 3>;
    using level_2 = level<0, 1, 3>;
    using level_3 = level<1, 1, 3>;

    static_assert(impl_::level_gt::apply<level_2, level_1>::value, "");
    static_assert(impl_::level_gt::apply<level_3, level_2>::value, "");
    static_assert(impl_::level_gt::apply<level_3, level_1>::value, "");
    static_assert(!impl_::level_gt::apply<level_1, level_2>::value, "");
    static_assert(!impl_::level_gt::apply<level_2, level_3>::value, "");
    static_assert(!impl_::level_gt::apply<level_1, level_3>::value, "");
    static_assert(!impl_::level_gt::apply<level_1, level_1>::value, "");
    static_assert(!impl_::level_gt::apply<level_2, level_2>::value, "");
    static_assert(!impl_::level_gt::apply<level_3, level_3>::value, "");
}
