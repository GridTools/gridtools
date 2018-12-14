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

#include <gridtools/stencil-composition/sid/composite.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil-composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        namespace tu = tuple_util;
        using tu::get;

        TEST(composite, deref) {
            double const src = 42;
            double dst = 0;

            auto testee = tu::make<sid::composite>(
                sid::synthetic().set<property::origin>(&src), sid::synthetic().set<property::origin>(&dst));
            static_assert(is_sid<decltype(testee)>(), "");

            auto ptrs = sid::get_origin(testee);
            EXPECT_EQ(&src, get<0>(ptrs));
            EXPECT_EQ(&dst, get<1>(ptrs));

            auto refs = *ptrs;
            get<1>(refs) = get<0>(refs);
            EXPECT_EQ(42, dst);
        }

        struct my_strides_kind;

        TEST(composite, functional) {
            double const one[5] = {0, 10, 20, 30, 40};
            double two = -1;
            double three[4][3][5] = {};
            char four[4][3][5] = {};

            auto my_strides = tu::make<array>(1, 5, 15);

            auto testee = tu::make<sid::composite>                                                    //
                (                                                                                     //
                    sid::synthetic()                                                                  //
                        .set<property::origin>(&one[0])                                               //
                        .set<property::strides>(tuple_util::make<tuple>(integral_constant<int, 1>())) //
                    ,                                                                                 //
                    sid::synthetic()                                                                  //
                        .set<property::origin>(&two)                                                  //
                    ,                                                                                 //
                    sid::synthetic()                                                                  //
                        .set<property::origin>(&three[0][0][0])                                       //
                        .set<property::strides>(my_strides)                                           //
                        .set<property::strides_kind, my_strides_kind>()                               //
                    ,                                                                                 //
                    sid::synthetic()                                                                  //
                        .set<property::origin>(&four[0][0][0])                                        //
                        .set<property::strides>(my_strides)                                           //
                        .set<property::strides_kind, my_strides_kind>()                               //
                );
            static_assert(is_sid<decltype(testee)>(), "");

            auto &&strides = sid::get_strides(testee);
            auto &&stride0 = get<0>(strides);

            EXPECT_EQ(1, get<0>(stride0));
            EXPECT_EQ(0, get<1>(stride0));

            auto ptr = sid::get_origin(testee);

            EXPECT_EQ(0, get<0>(*ptr));
            EXPECT_EQ(-1, get<1>(*ptr));
            EXPECT_EQ(&four[0][0][0], get<3>(ptr));

            using ptr_diff_t = GT_META_CALL(sid::ptr_diff_type, decltype(testee));

            ptr_diff_t ptr_diff;
            EXPECT_EQ(0, get<0>(ptr_diff));
            EXPECT_EQ(0, get<1>(ptr_diff));

            sid::shift(ptr_diff, stride0, 1);
            EXPECT_EQ(1, get<0>(ptr_diff));
            EXPECT_EQ(0, get<1>(ptr_diff));

            sid::shift(ptr_diff, stride0, integral_constant<int, 2>());
            EXPECT_EQ(3, get<0>(ptr_diff));
            EXPECT_EQ(0, get<1>(ptr_diff));

            ptr = ptr + ptr_diff;
            EXPECT_EQ(30, get<0>(*ptr));
            EXPECT_EQ(-1, get<1>(*ptr));

            *get<1>(ptr) = *get<0>(ptr);
            EXPECT_EQ(30, get<1>(*ptr));

            sid::shift(ptr, stride0, -2);
            EXPECT_EQ(10, get<0>(*ptr));
            EXPECT_EQ(30, get<1>(*ptr));

            EXPECT_EQ(&three[0][0][1], get<2>(ptr));
            EXPECT_EQ(&four[0][0][1], get<3>(ptr));

            sid::shift(ptr, get<1>(strides), 2);
            sid::shift(ptr, get<2>(strides), integral_constant<int, 3>());
            EXPECT_EQ(&three[3][2][1], get<2>(ptr));
            EXPECT_EQ(&four[3][2][1], get<3>(ptr));

            ptr_diff = {};
            sid::shift(ptr_diff, get<0>(strides), 3);
            sid::shift(ptr_diff, get<1>(strides), 2);
            sid::shift(ptr_diff, get<2>(strides), 1);
            ptr = sid::get_origin(testee) + ptr_diff;
            EXPECT_EQ(&three[1][2][3], get<2>(ptr));
            EXPECT_EQ(&four[1][2][3], get<3>(ptr));
        }
    } // namespace
} // namespace gridtools
