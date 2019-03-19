/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple_util.hpp>

#include <array>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/pair.hpp>
#include <gridtools/meta.hpp>

namespace custom {
    struct foo {
        int a;
        double b;

        struct getter {
            template <size_t I, gridtools::enable_if_t<I == 0, int> = 0>
            static constexpr int get(foo const &obj) {
                return obj.a;
            }
            template <size_t I, gridtools::enable_if_t<I == 0, int> = 0>
            static int &get(foo &obj) {
                return obj.a;
            }
            template <size_t I, gridtools::enable_if_t<I == 0, int> = 0>
            static constexpr int get(foo &&obj) {
                return obj.a;
            }
            template <size_t I, gridtools::enable_if_t<I == 1, int> = 0>
            static constexpr double get(foo const &obj) {
                return obj.b;
            }
            template <size_t I, gridtools::enable_if_t<I == 1, int> = 0>
            static double &get(foo &obj) {
                return obj.b;
            }
            template <size_t I, gridtools::enable_if_t<I == 1, int> = 0>
            static constexpr double get(foo &&obj) {
                return obj.b;
            }
        };
        friend getter tuple_getter(foo);
        friend gridtools::meta::list<int, double> tuple_to_types(foo);
        friend gridtools::meta::always<foo> tuple_from_types(foo);
    };
} // namespace custom

namespace gridtools {
    namespace tuple_util {
        TEST(get, std_tuple) {
            auto obj = std::make_tuple(1, 2.);
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        TEST(get, std_pair) {
            auto obj = std::make_pair(1, 2.);
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        TEST(get, std_array) {
            auto obj = make<std::array>(1, 2);
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);
        }

        struct add_2_f {
            template <class T>
            GT_FUNCTION constexpr T operator()(T val) const {
                return val + 2;
            }
        };

        TEST(get, custom) {
            custom::foo obj{1, 2};
            EXPECT_EQ(get<0>(obj), 1);
            EXPECT_EQ(get<1>(obj), 2);
            get<0>(obj) = 42;
            EXPECT_EQ(get<0>(obj), 42);

            constexpr custom::foo c_obj{2, 4};
            static_assert(get<0>(c_obj) == 2, "");
            static_assert(get<0>(custom::foo{3, 0.0}) == 3, "");
            static_assert(size<custom::foo>::value == 2, "");

            auto res = transform(add_2_f{}, custom::foo{42, 5.3});
            static_assert(std::is_same<decltype(res), custom::foo>{}, "");
            EXPECT_EQ(res.a, 44);
            EXPECT_EQ(res.b, 7.3);

            constexpr auto c_res = transform(add_2_f{}, custom::foo{42, 5.3});
            static_assert(c_res.a == 44, "");
            static_assert(c_res.b == 7.3, "");
        }

        TEST(transform, functional) {
            auto src = std::make_tuple(42, 5.3);
            auto res = transform(add_2_f{}, src);
            static_assert(std::is_same<decltype(res), decltype(src)>{}, "");
            EXPECT_EQ(res, std::make_tuple(44, 7.3));
        }

        TEST(transform, array) {
            auto src = make<std::array>(42, 5);
            auto res = transform(add_2_f{}, src);
            static_assert(std::is_same<decltype(res), decltype(src)>{}, "");
            EXPECT_THAT(res, testing::ElementsAre(44, 7));
        }

        TEST(transform, gt_array) {
            auto src = make<gridtools::array>(42, 5);
            auto res = host_device::transform(add_2_f{}, src);
            static_assert(std::is_same<decltype(res), decltype(src)>{}, "");
            EXPECT_THAT(res, testing::ElementsAre(44, 7));
        }

        TEST(transform, multiple_inputs) {
            EXPECT_EQ(std::make_tuple(11, 22),
                transform([](int lhs, int rhs) { return lhs + rhs; }, std::make_tuple(1, 2), std::make_tuple(10, 20)));
        }

        TEST(transform, multiple_arrays) {
            EXPECT_THAT(
                transform([](int lhs, int rhs) { return lhs + rhs; }, make<std::array>(1, 2), make<std::array>(10, 20)),
                testing::ElementsAre(11, 22));
        }

        struct accumulate_f {
            double &m_acc;
            template <class T>
            void operator()(T val) const {
                m_acc += val;
            }
        };

        TEST(for_each, functional) {
            double acc = 0;
            for_each(accumulate_f{acc}, std::make_tuple(42, 5.3));
            EXPECT_EQ(47.3, acc);
        }

        TEST(for_each, array) {
            double acc = 0;
            for_each(accumulate_f{acc}, make<std::array>(42, 5.3));
            EXPECT_EQ(47.3, acc);
        }

        TEST(for_each, multiple_inputs) {
            int acc = 0;
            for_each([&](int lhs, int rhs) { acc += lhs + rhs; }, std::make_tuple(1, 2), std::make_tuple(10, 20));
            EXPECT_EQ(33, acc);
        }

        struct accumulate2_f {
            double &m_acc;
            template <class T, class U>
            void operator()(T lhs, U rhs) const {
                m_acc += lhs * rhs;
            }
        };

        TEST(for_each_in_cartesian_product, functional) {
            double acc = 0;
            for_each_in_cartesian_product(accumulate2_f{acc}, std::make_tuple(1, 2), std::make_tuple(10, 20));
            EXPECT_EQ(90, acc);
        }

        TEST(for_each_in_cartesian_product, array) {
            double acc = 0;
            for_each_in_cartesian_product(accumulate2_f{acc}, make<std::array>(1, 2), make<std::array>(10, 20));
            EXPECT_EQ(90, acc);
        }

        TEST(flatten, functional) {
            EXPECT_EQ(
                flatten(std::make_tuple(std::make_tuple(1, 2), std::make_tuple(3, 4))), std::make_tuple(1, 2, 3, 4));
        }

        TEST(flatten, array) {
            EXPECT_THAT(flatten(std::make_tuple(make<std::array>(1, 2), make<std::array>(3, 4))),
                testing::ElementsAre(1, 2, 3, 4));
        }

        TEST(flatten, ref) {
            auto orig = std::make_tuple(std::make_tuple(1, 2), std::make_tuple(3, 4));
            auto flat = flatten(orig);
            EXPECT_EQ(flat, std::make_tuple(1, 2, 3, 4));
            get<0>(flat) = 42;
            EXPECT_EQ(get<0>(get<0>(orig)), 42);
        }

        TEST(drop_front, functional) { EXPECT_EQ(drop_front<2>(std::make_tuple(1, 2, 3, 4)), std::make_tuple(3, 4)); }

        TEST(drop_front, array) {
            EXPECT_THAT(drop_front<2>(make<std::array>(1, 2, 3, 4)), testing::ElementsAre(3, 4));
        }

        TEST(push_back, functional) { EXPECT_EQ(push_back(std::make_tuple(1, 2), 3, 4), std::make_tuple(1, 2, 3, 4)); }

        TEST(push_back, array) {
            EXPECT_THAT(push_back(make<std::array>(1, 2), 3, 4), testing::ElementsAre(1, 2, 3, 4));
        }

        TEST(push_front, functional) {
            EXPECT_EQ(push_front(std::make_tuple(1, 2), 3, 4), std::make_tuple(3, 4, 1, 2));
        }

        TEST(fold, functional) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, std::make_tuple(1, 2, 3, 4, 5, 6)), 21);
        }

        TEST(fold, with_state) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, 100, std::make_tuple(1, 2, 3, 4, 5, 6)), 121);
        }

        TEST(fold, array) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, make<std::array>(1, 2, 3, 4, 5, 6)), 21);
        }

        TEST(apply, lambda) {
            auto f = [](int x, int y) { return x + y; };
            auto t = std::make_tuple(1, 2);

            EXPECT_EQ(
                3, gridtools::tuple_util::apply(f, t)); // fully qualified to resolve ambiguity with c++17 std::apply
        }

        TEST(apply, array) {
            EXPECT_EQ(3,
                gridtools::tuple_util::apply([](int x, int y) { return x + y; },
                    make<std::array>(1, 2))); // fully qualified to resolve ambiguity with c++17 std::apply
        }

        TEST(make, functional) { EXPECT_EQ(make<std::tuple>(42, 5.3), std::make_tuple(42, 5.3)); }

        TEST(make, pair) { EXPECT_EQ(make<std::pair>(42, 5.3), std::make_pair(42, 5.3)); }

        TEST(tie, functional) {
            int x = 0, y = 0;
            tie<std::tuple>(x, y) = std::make_tuple(3, 4);
            EXPECT_EQ(3, x);
            EXPECT_EQ(4, y);
        }

        TEST(tie, pair) {
            int x = 0, y = 0;
            tie<std::pair>(x, y) = std::make_pair(3, 4);
            EXPECT_EQ(3, x);
            EXPECT_EQ(4, y);
        }

        TEST(make, array) {
            EXPECT_THAT(make<std::array>(3, 4), testing::ElementsAre(3, 4));
            EXPECT_THAT((make<std::array>(3.5, 4)), testing::ElementsAre(3.5, 4.));
            EXPECT_THAT((make<std::array, int>(3, 4.2)), testing::ElementsAre(3, 4));
        }

        TEST(convert_to, tuple) {
            EXPECT_EQ(make<std::tuple>(1, 2), convert_to<std::tuple>(make<std::array>(1, 2)));
            EXPECT_EQ(make<std::tuple>(1, 2), convert_to<std::tuple>(make<gridtools::pair>(1, 2)));
            EXPECT_EQ(make<std::pair>(1, 2), convert_to<std::pair>(make<gridtools::array>(1, 2)));
        }

        TEST(convert_to, array) {
            EXPECT_THAT(convert_to<std::array>()(make<std::tuple>(3, 4)), testing::ElementsAre(3, 4));
            EXPECT_THAT(convert_to<std::array>(make<std::tuple>(3, 4)), testing::ElementsAre(3, 4));
            EXPECT_THAT(convert_to<std::array>(make<std::tuple>(3.5, 4)), testing::ElementsAre(3.5, 4.));
            EXPECT_THAT((convert_to<std::array, int>(make<std::tuple>(3.5, 4))), testing::ElementsAre(3, 4));
            EXPECT_THAT((convert_to<std::array, int>()(make<std::tuple>(3.5, 4))), testing::ElementsAre(3, 4));
            EXPECT_THAT((convert_to<std::array, int>(make<std::array>(3.5, 4))), testing::ElementsAre(3, 4));
        }

        TEST(transpose, functional) {
            EXPECT_EQ(
                transpose(make<std::array>(make<std::pair>(1, 10), make<std::pair>(2, 20), make<std::pair>(3, 30))),
                make<std::pair>(make<std::array>(1, 2, 3), make<std::array>(10, 20, 30)));
        }

        TEST(all_of, functional) {
            auto testee = all_of([](int i) { return i % 2; });
            EXPECT_TRUE(testee(make<std::tuple>(1, 3, 99, 7)));
            EXPECT_FALSE(testee(make<std::tuple>(1, 3, 2, 7, 100)));

            EXPECT_TRUE(all_of(
                [](int l, int r) { return l == r; }, make<std::tuple>(1, 3, 99, 7), make<std::tuple>(1, 3, 99, 7)));
        }

        TEST(reverse, functional) {
            EXPECT_TRUE(reverse(make<std::tuple>(1, 'a', 42.5)) == make<std::tuple>(42.5, 'a', 1));
        }
    } // namespace tuple_util
} // namespace gridtools
