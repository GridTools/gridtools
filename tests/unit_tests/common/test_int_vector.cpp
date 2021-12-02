/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gridtools/common/array.hpp"
#include "gridtools/meta/debug.hpp"
#include <gridtools/common/int_vector.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
    namespace {
        using namespace literals;

        struct a;
        struct b;
        struct c;

        TEST(plus, integrals) {
            auto m1 = tuple_util::make<hymap::keys<a, b>::values>(1, 2l);
            auto m2 = tuple_util::make<hymap::keys<b, c>::values>(10, 20);
            auto m3 = tuple_util::make<hymap::keys<c, b>::values>(5u, 6);
            auto testee = int_vector::plus(m1, m2, m3);

            static_assert(std::is_same_v<int, std::decay_t<decltype(at_key<a>(testee))>>);
            static_assert(std::is_same_v<long int, std::decay_t<decltype(at_key<b>(testee))>>);
            static_assert(std::is_same_v<unsigned int, std::decay_t<decltype(at_key<c>(testee))>>);

            EXPECT_EQ(1, at_key<a>(testee));
            EXPECT_EQ(18, at_key<b>(testee));
            EXPECT_EQ(25, at_key<c>(testee));
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
        }

        TEST(multiply, integral_constants) {
            auto vec = tuple_util::make<hymap::keys<a, b>::values>(integral_constant<int, 1>{}, 2);

            auto testee = int_vector::multiply(vec, integral_constant<int, 2>{});

            static_assert(std::is_same_v<integral_constant<int, 2>, std::decay_t<decltype(at_key<a>(testee))>>);
            EXPECT_EQ(4, at_key<b>(testee));
        }

    } // namespace
} // namespace gridtools
