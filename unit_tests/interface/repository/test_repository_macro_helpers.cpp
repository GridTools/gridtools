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
#include "interface/repository/repository_macro_helpers.hpp"

TEST(repository_macros, max_in_tuple) {
#define my_tuple (0, 1, 4)
    ASSERT_EQ(4, GTREPO_max_in_tuple(my_tuple));
#undef my_tuple
}

TEST(repository_macros, max_dim) {
#define my_field_types (IJKDataStore, (0, 1, 5))(IJDataStore, (0, 1))(AnotherDataStore, (8, 1))
    int result = GTREPO_max_dim(GRIDTOOLS_PP_SEQ_DOUBLE_PARENS(my_field_types));
    ASSERT_EQ(8, result);
#undef my_field_types
}

TEST(repository_macros, has_dim) {
#define my_field_types (IJKDataStore, (0, 1, 5))(IJDataStore, (0, 1))(AnotherDataStore, (8, 1))
    ASSERT_GT(GTREPO_has_dim(GRIDTOOLS_PP_SEQ_DOUBLE_PARENS(my_field_types)), 0);
#undef my_field_types
#define my_field_types (IJKDataStore)(IJDataStore)(AnotherDataStore)
    ASSERT_EQ(0, GTREPO_has_dim(GRIDTOOLS_PP_SEQ_DOUBLE_PARENS(my_field_types)));
#undef my_field_types
}
