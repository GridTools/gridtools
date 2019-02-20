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

#include <gridtools/common/hymap.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {

        struct a;
        struct b;
        struct c;

        TEST(tuple_like, smoke) {
            using testee_t = hymap_ctor<std::tuple>::keys<a, b, c>::values<int, double, void *>;

            static_assert(tuple_util::size<testee_t>::value == 3, "");

            static_assert(std::is_same<GT_META_CALL(tuple_util::element, (0, testee_t)), int>::value, "");
            static_assert(std::is_same<GT_META_CALL(tuple_util::element, (1, testee_t)), double>::value, "");
            static_assert(std::is_same<GT_META_CALL(tuple_util::element, (2, testee_t)), void *>::value, "");

            testee_t testee{42, 5.3, nullptr};
            EXPECT_EQ(42, tuple_util::get<0>(testee));
            EXPECT_EQ(5.3, tuple_util::get<1>(testee));
            EXPECT_EQ(nullptr, tuple_util::get<2>(testee));
        }

        TEST(at_key, smoke) {
            using testee_t = hymap_ctor<std::tuple>::keys<a, b>::values<int, double>;
            testee_t testee{42, 5.3};

            static_assert(has_key<testee_t, a>::value, "");
            static_assert(has_key<testee_t, b>::value, "");
            static_assert(!has_key<testee_t, c>::value, "");

            EXPECT_EQ(42, at_key<a>(testee));
            EXPECT_EQ(5.3, at_key<b>(testee));
        }

        TEST(at_key, tuple_like) {
            using testee_t = std::tuple<int, double>;
            testee_t testee{42, 5.3};

            static_assert(has_key<testee_t, integral_constant<int, 0>>::value, "");
            static_assert(has_key<testee_t, integral_constant<int, 1>>::value, "");
            static_assert(!has_key<testee_t, integral_constant<int, 2>>::value, "");

            EXPECT_EQ(42, (at_key<integral_constant<int, 0>>(testee)));
            EXPECT_EQ(5.3, (at_key<integral_constant<int, 1>>(testee)));
        }

        struct add_2_f {
            template <class T>
            T operator()(T x) const {
                return x + 2;
            }
        };

        TEST(tuple_like, transform) {
            using testee_t = hymap_ctor<std::tuple>::keys<a, b>::values<int, double>;

            testee_t src = {42, 5.3};
            auto dst = tuple_util::transform(add_2_f{}, src);

            EXPECT_EQ(44, at_key<a>(dst));
            EXPECT_EQ(7.3, at_key<b>(dst));
        }

        TEST(make_hymap, smoke) {
            auto testee = tuple_util::make<hymap_ctor<std::tuple>::keys<a, b>::values>(42, 5.3);

            EXPECT_EQ(42, at_key<a>(testee));
            EXPECT_EQ(5.3, at_key<b>(testee));
        }

    } // namespace
} // namespace gridtools