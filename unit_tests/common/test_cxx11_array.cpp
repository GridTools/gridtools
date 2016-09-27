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
#include "common/array.hpp"
#include "common/array_addons.hpp"

using namespace gridtools;

TEST(array, test_append) {
    array< uint_t, 4 > a{1, 2, 3, 4};
    auto mod_a = a.append_dim(5);
    ASSERT_TRUE((mod_a == array< uint_t, 5 >{1, 2, 3, 4, 5}));
    ASSERT_TRUE((mod_a[4] == 5));
}

TEST(array, test_prepend) {
    constexpr array< uint_t, 4 > a{1, 2, 3, 4};
    auto mod_a = a.prepend_dim(5);
    ASSERT_TRUE((mod_a == array< uint_t, 5 >{5, 1, 2, 3, 4}));
    ASSERT_TRUE((mod_a[0] == 5));
}

TEST(array, test_copyctr) {
    constexpr array< uint_t, 4 > a{4, 2, 3, 1};
    constexpr auto mod_a(a);
    ASSERT_TRUE((mod_a == array< uint_t, 4 >{4, 2, 3, 1}));
    ASSERT_TRUE((mod_a[0] == 4));
}
