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
            static_assert(element_at<a, testee>::value == -1);
        } // namespace extents_offsets

        namespace extents_sizes {
            using testee = decltype(extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>::sizes());

            static_assert(!has_key<testee, c>::value);
            static_assert(element_at<a, testee>::value == 1);
            static_assert(element_at<b, testee>::value == 2);
        } // namespace extents_sizes

        TEST(extents, extend_offsets) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto offsets = tuple_util::make<hymap::keys<a, b, c>::values>(0, 1, 2);

            auto testee = extend_offsets<ext>(offsets);

            EXPECT_EQ(at_key<a>(testee), -1);
            EXPECT_EQ(at_key<b>(testee), 1);
            EXPECT_EQ(at_key<c>(testee), 3);
        }

        TEST(extents, extend_sizes) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto sizes = tuple_util::make<hymap::keys<a, b, c>::values>(4, 5, 6);

            auto testee = extend_sizes<ext>(sizes);

            EXPECT_EQ(at_key<a>(testee), 5);
            EXPECT_EQ(at_key<b>(testee), 7);
            EXPECT_EQ(at_key<c>(testee), 6);
        }

        namespace extends_make_extends {
            using testee = make_extents<extent<a, -1, 1>, extent<b, -1, 1>, extent<a, -2, 0>, extent<b, 0, 3>>;

            static_assert(element_at<a, decltype(testee::offsets())>::value == -2);
            static_assert(element_at<a, decltype(testee::sizes())>::value == 3);
            static_assert(element_at<b, decltype(testee::offsets())>::value == -1);
            static_assert(element_at<b, decltype(testee::sizes())>::value == 4);
        } // namespace extends_make_extends

        namespace extents_enclosing_extents {
            using foo = extents<extent<a, -1, 1>, extent<b, -1, 1>>;
            using bar = extents<extent<a, -2, 0>, extent<b, 0, 3>>;
            using testee = enclosing_extents<foo, bar>;

            static_assert(element_at<a, decltype(testee::offsets())>::value == -2);
            static_assert(element_at<a, decltype(testee::sizes())>::value == 3);
            static_assert(element_at<b, decltype(testee::offsets())>::value == -1);
            static_assert(element_at<b, decltype(testee::sizes())>::value == 4);
        } // namespace extents_enclosing_extents

    } // namespace
} // namespace gridtools::fn
