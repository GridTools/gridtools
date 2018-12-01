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

#include <gridtools/common/tuple.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/macros.hpp>

namespace gridtools {
    namespace {

        template <size_t I>
        struct an_empty {
            static constexpr size_t value = I;
        };

        struct move_only {
            int value;

            move_only() = default;

            move_only(move_only const &) = delete;
            move_only(move_only &&) = default;

            move_only &operator=(move_only const &) = delete;
            move_only &operator=(move_only &&) = default;
        };

        struct take_move_only {
            int value;
            take_move_only() = default;
            GT_FUNCTION take_move_only(move_only src) : value(src.value) {}
        };

        template <size_t I>
        const size_t an_empty<I>::value;

        // empty base optimization works
        static_assert(sizeof(tuple<an_empty<0>, an_empty<1>, an_empty<3>>) == sizeof(an_empty<0>), "");

        static_assert(tuple_util::size<tuple<int, char, double, char>>() == 4, "");
        static_assert(tuple_util::size<tuple<int>>() == 1, "");
        static_assert(tuple_util::size<tuple<>>() == 0, "");

        static_assert(std::is_same<GT_META_CALL(tuple_util::element, (1, tuple<int, char, double, char>)), char>(), "");
        static_assert(std::is_same<GT_META_CALL(tuple_util::element, (0, tuple<int>)), int>(), "");

        using tuple_util::host_device::get;

        TEST(tuple, get) {
            tuple<int, double, an_empty<59>> testee;
            EXPECT_EQ(0, get<0>(testee));
            EXPECT_EQ(0, get<1>(testee));
            EXPECT_EQ(59, get<2>(testee).value);

            get<0>(testee) = 42;
            get<1>(testee) = get<0>(testee);
            get<2>(testee) = {};

            EXPECT_EQ(42, get<0>(testee));
            EXPECT_EQ(42, get<1>(testee));
            EXPECT_EQ(59, get<2>(testee).value);
        }

        TEST(one_tuple, get) {
            tuple<int> testee;
            EXPECT_EQ(0, get<0>(testee));

            get<0>(testee) = 42;

            EXPECT_EQ(42, get<0>(testee));
        }

        TEST(tuple, move_get) {
            auto val = get<1>(tuple<char, move_only>{'a', move_only{2}});
            static_assert(std::is_same<decltype(val), move_only>(), "");
            EXPECT_EQ(2, val.value);
        }

        TEST(one_tuple, move_get) {
            auto val = get<0>(tuple<move_only>{move_only{2}});
            static_assert(std::is_same<decltype(val), move_only>(), "");
            EXPECT_EQ(2, val.value);
        }

        TEST(tuple, element_wise_ctor) {
            int const v1 = 3;
            double const v2 = 2.5;
            an_empty<59> const v3 = {};
            tuple<int, double, an_empty<59>> testee{v1, v2, v3};
            EXPECT_EQ(3, get<0>(testee));
            EXPECT_EQ(2.5, get<1>(testee));
            EXPECT_EQ(59, get<2>(testee).value);
        }

        TEST(one_tuple, element_wise_ctor) {
            int const v = 3;
            tuple<int> testee{v};
            EXPECT_EQ(3, get<0>(testee));
        }

        TEST(tuple, copy_ctor) {
            tuple<int, double, an_empty<59>> src{3, 2.5, {}};
            auto testee = src;
            EXPECT_EQ(get<0>(src), get<0>(testee));
            EXPECT_EQ(get<1>(src), get<1>(testee));
            EXPECT_EQ(get<2>(src).value, get<2>(testee).value);
        }

        TEST(one_tuple, copy_ctor) {
            tuple<int> src{3};
            auto testee = src;
            EXPECT_EQ(get<0>(src), get<0>(testee));
        }

        TEST(tuple, move_element_wise_ctor) {
            tuple<move_only, move_only> testee{move_only{47}, move_only{2}};
            EXPECT_EQ(47, get<0>(testee).value);
            EXPECT_EQ(2, get<1>(testee).value);
        }

        TEST(one_tuple, move_element_wise_ctor) {
            tuple<move_only> testee{move_only{47}};
            EXPECT_EQ(47, get<0>(testee).value);
        }

        TEST(tuple, move_ctor) {
            auto testee = tuple<move_only, move_only>{move_only{47}, move_only{2}};
            EXPECT_EQ(47, get<0>(testee).value);
            EXPECT_EQ(2, get<1>(testee).value);
        }

        TEST(one_tuple, move_ctor) {
            auto testee = tuple<move_only>{move_only{47}};
            EXPECT_EQ(47, get<0>(testee).value);
        }

        TEST(tuple, element_wise_conversion_ctor) {
            tuple<int, double> testee{'a', 'b'};
            EXPECT_EQ('a', get<0>(testee));
            EXPECT_EQ('b', get<1>(testee));
        }

        TEST(one_tuple, element_wise_conversion_ctor) {
            tuple<int> testee{'a'};
            EXPECT_EQ('a', get<0>(testee));
        }

        TEST(tuple, tuple_conversion_copy_ctor) {
            tuple<char, char> src{'a', 'b'};
            tuple<int, double> testee = src;
            EXPECT_EQ('a', get<0>(testee));
            EXPECT_EQ('b', get<1>(testee));
        }

        TEST(one_tuple, tuple_conversion_copy_ctor) {
            tuple<char> src{'a'};
            tuple<int> testee = src;
            EXPECT_EQ('a', get<0>(testee));
        }

        TEST(tuple, tuple_conversion_copy_ctor_nested) {
            tuple<tuple<int, int>, int> src{{1, 2}, 3};
            tuple<tuple<double, double>, double> testee = src;
            EXPECT_EQ(1, get<0>(get<0>(testee)));
            EXPECT_EQ(2, get<1>(get<0>(testee)));
            EXPECT_EQ(3, get<1>(testee));
        }

        TEST(one_tuple, tuple_conversion_copy_ctor_nested) {
            tuple<tuple<int>> const src{{1}};
            tuple<tuple<double>> testee = src;
            EXPECT_EQ(1, get<0>(get<0>(testee)));
        }

        TEST(tuple, tuple_conversion_move_ctor) {
            tuple<double, move_only> testee = tuple<char, move_only>{'a', move_only{2}};
            EXPECT_EQ('a', get<0>(testee));
            EXPECT_EQ(2, get<1>(testee).value);
        }

        TEST(one_tuple, tuple_conversion_move_ctor) {
            tuple<take_move_only> testee = tuple<move_only>{move_only{2}};
            EXPECT_EQ(2, get<0>(testee).value);
        }

        TEST(tuple, copy_assign) {
            tuple<int, double> src = {1, 1.5};
            tuple<int, double> testee;
            auto &res = testee = src;
            static_assert(std::is_same<decltype(res), tuple<int, double> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ(1, get<0>(testee));
            EXPECT_EQ(1.5, get<1>(testee));
        }

        TEST(one_tuple, copy_assign) {
            tuple<int> src = {1};
            tuple<int> testee;
            auto &res = testee = src;
            static_assert(std::is_same<decltype(res), tuple<int> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ(1, get<0>(testee));
        }

        TEST(tuple, move_assign) {
            tuple<move_only, move_only> testee;
            auto &res = testee = tuple<move_only, move_only>{move_only{47}, move_only{2}};
            static_assert(std::is_same<decltype(res), tuple<move_only, move_only> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ(47, get<0>(testee).value);
            EXPECT_EQ(2, get<1>(testee).value);
        }

        TEST(one_tuple, move_assign) {
            tuple<take_move_only> testee;
            auto &res = testee = tuple<move_only>{move_only{47}};
            static_assert(std::is_same<decltype(res), tuple<take_move_only> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ(47, get<0>(testee).value);
        }

        TEST(tuple, copy_conversion_assign) {
            tuple<char, char> src = {'a', 'b'};
            tuple<int, double> testee;
            auto &res = testee = src;
            static_assert(std::is_same<decltype(res), tuple<int, double> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ('a', get<0>(testee));
            EXPECT_EQ('b', get<1>(testee));
        }

        TEST(one_tuple, copy_conversion_assign) {
            tuple<char> src = {'a'};
            tuple<int> testee;
            auto &res = testee = src;
            static_assert(std::is_same<decltype(res), tuple<int> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ('a', get<0>(testee));
        }

        TEST(tuple, move_conversion_assign) {
            tuple<double, move_only> testee;
            auto &res = testee = tuple<char, move_only>{'a', move_only{2}};
            static_assert(std::is_same<decltype(res), tuple<double, move_only> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ('a', get<0>(testee));
            EXPECT_EQ(2, get<1>(testee).value);
        }

        TEST(one_tuple, move_conversion_assign) {
            tuple<take_move_only> testee;
            auto &res = testee = tuple<move_only>{move_only{2}};
            static_assert(std::is_same<decltype(res), tuple<take_move_only> &>(), "");
            EXPECT_EQ(&testee, &res);
            EXPECT_EQ(2, get<0>(testee).value);
        }

        TEST(tuple, swap_method) {
            tuple<int, double> a{1, 2}, b{10, 20};
            a.swap(b);
            EXPECT_EQ(10, get<0>(a));
            EXPECT_EQ(20, get<1>(a));
            EXPECT_EQ(1, get<0>(b));
            EXPECT_EQ(2, get<1>(b));
        }

        TEST(one_tuple, swap_method) {
            tuple<int> a{1}, b{10};
            a.swap(b);
            EXPECT_EQ(10, get<0>(a));
            EXPECT_EQ(1, get<0>(b));
        }

        TEST(tuple, swap) {
            tuple<int, double> a{1, 2}, b{10, 20};
            swap(a, b);
            EXPECT_EQ(10, get<0>(a));
            EXPECT_EQ(20, get<1>(a));
            EXPECT_EQ(1, get<0>(b));
            EXPECT_EQ(2, get<1>(b));
        }

        TEST(one_tuple, swap) {
            tuple<int> a{1}, b{10};
            swap(a, b);
            EXPECT_EQ(10, get<0>(a));
            EXPECT_EQ(1, get<0>(b));
        }

        TEST(empty_tuple, functional) {
            tuple<> src;
            tuple<> dst;

            auto copy = src;
            static_assert(std::is_same<decltype(copy), tuple<>>(), "");
            EXPECT_NE(&src, &copy);

            auto move = tuple<>{};
            static_assert(std::is_same<decltype(move), tuple<>>(), "");
            // make nvcc happy
            EXPECT_EQ(&move, &move);

            auto &copy_assign = dst = src;
            static_assert(std::is_same<decltype(copy_assign), tuple<> &>(), "");
            EXPECT_EQ(&dst, &copy_assign);

            auto &move_assign = dst = tuple<>{};
            static_assert(std::is_same<decltype(move_assign), tuple<> &>(), "");
            EXPECT_EQ(&dst, &move_assign);

            src.swap(dst);

            swap(src, dst);
        }

    } // namespace
} // namespace gridtools
