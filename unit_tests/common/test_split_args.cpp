/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/split_args.hpp>

#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

namespace gridtools {

    template <class Testee, class FirstSink, class SecondSink>
    void helper(Testee &&testee, FirstSink first_sink, SecondSink second_sink) {
        first_sink(std::forward<Testee>(testee).first);
        second_sink(std::forward<Testee>(testee).second);
    }

    TEST(raw_split_args, functional) {
        int val = 1;
        const int c_val = 2;
        helper(raw_split_args<std::is_lvalue_reference>(42, c_val, 0., val, c_val),
            [](std::tuple<int const &, int &, int const &> const &x) { EXPECT_EQ(std::make_tuple(2, 1, 2), x); },
            [](std::tuple<int &&, double &&> const &x) { EXPECT_EQ(std::make_tuple(42, 0.), x); });
    }

    TEST(split_args, functional) {
        int ival = 1;
        const double dval = 2;
        helper(split_args<std::is_integral>(42, dval, 0., ival),
            [](std::tuple<int &&, int &> const &x) { EXPECT_EQ(std::make_tuple(42, 1), x); },
            [](std::tuple<const double &, double &&> const &x) { EXPECT_EQ(std::make_tuple(2., 0.), x); });
    }
} // namespace gridtools
