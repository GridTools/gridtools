/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/int_vector.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {
        struct a;
        struct b;
        struct c;

        TEST(plus, integrals) {
            auto m1 = tuple_util::make<hymap::keys<a, b>::values>(1, 2l);
            auto m2 = tuple_util::make<hymap::keys<b, c>::values>(10, 20u);
            auto m3 = tuple_util::make<hymap::keys<b>::values>(100);

            auto testee = int_vector::plus(m1, m2, m3);

            static_assert(std::is_same_v<int, std::decay_t<decltype(at_key<a>(testee))>>);
            static_assert(std::is_same_v<long int, std::decay_t<decltype(at_key<b>(testee))>>);
            static_assert(std::is_same_v<unsigned int, std::decay_t<decltype(at_key<c>(testee))>>);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_EQ(112, at_key<b>(testee));
            EXPECT_EQ(20, at_key<c>(testee));

            using namespace int_vector::arithmetic;
            auto testee2 = m1 + m2;
            EXPECT_EQ(1, at_key<a>(testee2));
            EXPECT_EQ(12, at_key<b>(testee2));
            EXPECT_EQ(20, at_key<c>(testee2));
        }

        TEST(plus, integral_constants) {
            auto m1 = hymap::keys<a, b>::values<integral_constant<int, 1>, integral_constant<int, 2>>{};
            auto m2 = tuple_util::make<hymap::keys<a, b>::values>(integral_constant<int, 11>{}, 12);

            auto testee = int_vector::plus(m1, m2);

            static_assert(std::is_same_v<integral_constant<int, 12>, std::decay_t<decltype(at_key<a>(testee))>>);
            static_assert(std::is_same_v<int, std::decay_t<decltype(at_key<b>(testee))>>);

            EXPECT_EQ(14, at_key<b>(testee));
        }

        TEST(plus, tuple_and_arrays) {
            auto m1 = tuple<int, int>{1, 2};
            auto m2 = array<int, 2>{3, 4};
            auto testee = int_vector::plus(m1, m2);

            EXPECT_EQ(4, (at_key<integral_constant<int, 0>>(testee)));
            EXPECT_EQ(6, (at_key<integral_constant<int, 1>>(testee)));
        }

        TEST(multiply, integrals) {
            auto vec = tuple_util::make<hymap::keys<a, b>::values>(integral_constant<int, 1>{}, 2);

            auto testee = int_vector::multiply(vec, 2);

            EXPECT_EQ(2, at_key<a>(testee));
            EXPECT_EQ(4, at_key<b>(testee));

            using namespace int_vector::arithmetic;
            auto testee2 = vec * 2;
            EXPECT_EQ(2, at_key<a>(testee2));
            EXPECT_EQ(4, at_key<b>(testee2));

            auto testee3 = 2 * vec;
            EXPECT_EQ(2, at_key<a>(testee3));
            EXPECT_EQ(4, at_key<b>(testee3));
        }

        TEST(multiply, integral_constants) {
            auto vec = tuple_util::make<hymap::keys<a, b>::values>(integral_constant<int, 1>{}, 2);

            auto testee = int_vector::multiply(vec, integral_constant<int, 2>{});

            static_assert(std::is_same_v<integral_constant<int, 2>, std::decay_t<decltype(at_key<a>(testee))>>);
            EXPECT_EQ(4, at_key<b>(testee));
        }

        TEST(prune_zeros, smoke) {
            auto vec = tuple_util::make<hymap::keys<a, b, c>::values>(
                1, integral_constant<int, 0>{}, integral_constant<int, 2>{});

            auto testee = int_vector::prune_zeros(vec);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_FALSE((has_key<decltype(testee), b>{}));
            static_assert(std::is_same_v<integral_constant<int, 2>, std::decay_t<decltype(at_key<c>(testee))>>);
        }

        TEST(unary_ops, smoke) {
            using namespace int_vector::arithmetic;

            auto vec = tuple_util::make<hymap::keys<a, b, c>::values>(
                1, integral_constant<int, 0>{}, integral_constant<int, 2>{});

            auto testee = -vec;

            EXPECT_EQ(-1, at_key<a>(testee));
            static_assert(std::is_same_v<integral_constant<int, 0>, std::decay_t<decltype(at_key<b>(testee))>>);
            static_assert(std::is_same_v<integral_constant<int, -2>, std::decay_t<decltype(at_key<c>(testee))>>);

            auto testee2 = +vec;
            EXPECT_EQ(1, at_key<a>(testee2));
            static_assert(std::is_same_v<integral_constant<int, 0>, std::decay_t<decltype(at_key<b>(testee2))>>);
            static_assert(std::is_same_v<integral_constant<int, 2>, std::decay_t<decltype(at_key<c>(testee2))>>);
        }

        TEST(minus_op, smoke) {
            using namespace int_vector::arithmetic;

            auto m1 = tuple_util::make<hymap::keys<a, b>::values>(1, integral_constant<int, 2>{});
            auto m2 = tuple_util::make<hymap::keys<a, b, c>::values>(1, integral_constant<int, 1>{}, 3);

            auto testee = m1 - m2;

            EXPECT_EQ(0, at_key<a>(testee));
            static_assert(std::is_same_v<integral_constant<int, 1>, std::decay_t<decltype(at_key<b>(testee))>>);
            EXPECT_EQ(-3, at_key<c>(testee));
        }

    } // namespace
} // namespace gridtools
