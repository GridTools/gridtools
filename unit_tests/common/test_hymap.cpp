/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
            using testee_t = hymap::keys<a, b, c>::values<int, double, void *>;

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
            using testee_t = hymap::keys<a, b>::values<int, double>;
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
            using testee_t = hymap::keys<a, b>::values<int, double>;

            testee_t src = {42, 5.3};
            auto dst = tuple_util::transform(add_2_f{}, src);

            EXPECT_EQ(44, at_key<a>(dst));
            EXPECT_EQ(7.3, at_key<b>(dst));
        }

        TEST(make_hymap, smoke) {
            auto testee = tuple_util::make<hymap::keys<a, b>::values>(42, 5.3);

            EXPECT_EQ(42, at_key<a>(testee));
            EXPECT_EQ(5.3, at_key<b>(testee));
        }

        TEST(convert_hymap, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 5.3};

            auto dst = tuple_util::convert_to<hymap::keys<b, c>::values>(src);

            EXPECT_EQ(42, at_key<b>(dst));
            EXPECT_EQ(5.3, at_key<c>(dst));
        }

        TEST(to_meta_map, smoke) {
            using src_t = hymap::keys<a, b>::values<int, double>;
            using dst_t = GT_META_CALL(hymap::to_meta_map, src_t);

            static_assert(
                std::is_same<GT_META_CALL(meta::second, (GT_META_CALL(meta::mp_find, (dst_t, a)))), int>(), "");
            static_assert(
                std::is_same<GT_META_CALL(meta::second, (GT_META_CALL(meta::mp_find, (dst_t, b)))), double>(), "");
        }

        TEST(from_meta_map, smoke) {
            using src_t = meta::list<meta::list<a, int>, meta::list<b, double>>;
            using dst_t = GT_META_CALL(hymap::from_meta_map, src_t);

            static_assert(
                std::is_same<typename std::decay<decltype(at_key<a>(std::declval<dst_t>()))>::type, int>(), "");
            static_assert(
                std::is_same<typename std::decay<decltype(at_key<b>(std::declval<dst_t>()))>::type, double>(), "");
        }

        struct add_3_f {
            template <class Key, class Value>
            Value operator()(Value x) const {
                return x + 3;
            }
        };

        TEST(transform, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 5.3};

            auto dst = hymap::transform(add_3_f{}, src);

            EXPECT_EQ(45, at_key<a>(dst));
            EXPECT_EQ(8.3, at_key<b>(dst));
        }

        struct acc_f {
            double &m_val;
            template <class Key, class Value>
            void operator()(Value x) const {
                m_val += x;
            }
        };

        TEST(for_each, smoke) {
            hymap::keys<a, b>::values<int, double> src = {42, 5.3};

            double acc = 0;
            hymap::for_each(acc_f{acc}, src);

            EXPECT_EQ(47.3, acc);
        }
    } // namespace
} // namespace gridtools
