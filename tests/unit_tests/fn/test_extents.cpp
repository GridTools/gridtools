/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/fn/extents.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>

namespace gridtools::fn {
    namespace {
        template <class A,
            class B,
            class CommonValueType = std::common_type_t<typename A::value_type, typename B::value_type>>
        struct integral_constant_equal
            : std::is_same<integral_constant<CommonValueType, A::value>, integral_constant<CommonValueType, B::value>> {
        };

        using namespace literals;
        struct a {};
        struct b {};
        struct c {};

        static_assert(is_extent<extent<a, 0, 0>>::value);
        static_assert(is_extent<extent<a, -1, 1>>::value);

        namespace {
            using testee = extents<extent<a, 0, 0>, extent<b, 0, 0>>;
        }

        namespace extents_offsets {
            using testee = decltype(extents<extent<a, -1, 0>, extent<b, 0, 2>>::offsets());

            static_assert(!has_key<testee, b>::value);
            static_assert(integral_constant_equal<decltype(at_key<a>(std::declval<testee>())), decltype(-1_c)>::value);
        } // namespace extents_offsets

        namespace extents_sizes {
            using testee = decltype(extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>::sizes());

            static_assert(!has_key<testee, c>::value);
            static_assert(integral_constant_equal<decltype(at_key<a>(std::declval<testee>())), decltype(1_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<b>(std::declval<testee>())), decltype(2_c)>::value);
        } // namespace extents_sizes

        TEST(extents, extend_offsets) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto offsets = tuple_util::make<hymap::keys<a, b, c>::values>(0, 1, 2);

            auto testee = extend_offsets<ext>(offsets);

            ASSERT_EQ(at_key<a>(testee), -1);
            ASSERT_EQ(at_key<b>(testee), 1);
            ASSERT_EQ(at_key<c>(testee), 3);
        }

        TEST(extents, extend_sizes) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto sizes = tuple_util::make<hymap::keys<a, b, c>::values>(4, 5, 6);

            auto testee = extend_sizes<ext>(sizes);

            ASSERT_EQ(at_key<a>(testee), 5);
            ASSERT_EQ(at_key<b>(testee), 7);
            ASSERT_EQ(at_key<c>(testee), 6);
        }

        namespace extends_make_extends {
            using testee = make_extents<extent<a, -1, 1>, extent<b, -1, 1>, extent<a, -2, 0>, extent<b, 0, 3>>;

            static_assert(integral_constant_equal<decltype(at_key<a>(testee::offsets())), decltype(-2_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<a>(testee::sizes())), decltype(3_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<b>(testee::offsets())), decltype(-1_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<b>(testee::sizes())), decltype(4_c)>::value);
        } // namespace extends_make_extends

        namespace extents_enclosing_extents {
            using foo = extents<extent<a, -1, 1>, extent<b, -1, 1>>;
            using bar = extents<extent<a, -2, 0>, extent<b, 0, 3>>;
            using testee = enclosing_extents<foo, bar>;

            static_assert(integral_constant_equal<decltype(at_key<a>(testee::offsets())), decltype(-2_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<a>(testee::sizes())), decltype(3_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<b>(testee::offsets())), decltype(-1_c)>::value);
            static_assert(integral_constant_equal<decltype(at_key<b>(testee::sizes())), decltype(4_c)>::value);
        } // namespace extents_enclosing_extents

    } // namespace
} // namespace gridtools::fn
