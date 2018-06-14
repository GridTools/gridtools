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

#include <gridtools/common/any_moveable.hpp>

#include <memory>
#include <gtest/gtest.h>

namespace gridtools {

    TEST(any_moveable, smoke) {
        any_moveable x = 42;
        EXPECT_TRUE(x.has_value());
        EXPECT_EQ(typeid(int), x.type());
        EXPECT_EQ(42, any_cast< int >(x));
        auto &ref = any_cast< int & >(x);
        ref = 88;
        EXPECT_EQ(88, any_cast< int >(x));
        EXPECT_FALSE(any_cast< double * >(&x));
    }

    TEST(any_moveable, empty) { EXPECT_FALSE(any_moveable{}.has_value()); }

    TEST(any_moveable, move_only) {
        using testee_t = std::unique_ptr< int >;
        any_moveable x = testee_t(new int(42));
        EXPECT_EQ(42, *any_cast< testee_t const & >(x));
        any_moveable y = std::move(x);
        EXPECT_EQ(42, *any_cast< testee_t const & >(y));
    }
}
