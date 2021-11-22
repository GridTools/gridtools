/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdlib>
#include <utility>

#include <gtest/gtest.h>

#include <gridtools/fn.hpp>
#include <gridtools/fn/backend/naive.hpp>

using namespace gridtools;
using namespace literals;
using namespace fn;

template <auto D>
constexpr auto ldif = [](auto const &in) { return minus(deref(in), deref(shift<D, -1>(in))); };

template <auto D>
constexpr auto rdif = [](auto const &in) { return lambda<ldif<D>>(shift<D, 1>(in)); };

template <auto D, bool UseTmp>
constexpr auto d2 = [](auto const &in) { return lambda<ldif<D>>(lift<rdif<D>, UseTmp>(in)); };

template <class Param>
constexpr auto lap = [](auto const &in) { return plus(Param::d2i(in), Param::d2j(in)); };

template <bool I, bool J>
struct param {
    static constexpr auto d2i = lambda<d2<i, I>>;
    static constexpr auto d2j = lambda<d2<j, J>>;
};

using params_t = testing::Types<param<false, false>, param<true, false>, param<false, true>, param<true, true>>;

template <class>
using lift_test = testing::Test;

TYPED_TEST_SUITE(lift_test, params_t);

TYPED_TEST(lift_test, lap) {
    double actual[10][10][3] = {};
    double in[10][10][3];
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 3; ++k)
                in[i][j][k] = std::rand();

    auto expected = [&](auto i, auto j, auto k) {
        return in[i + 1][j][k] + in[i - 1][j][k] + in[i][j + 1][k] + in[i][j - 1][k] - 4 * in[i][j][k];
    };

    using stage_t = make_stage<lap<TypeParam>, std::identity{}, 0, 1>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = cartesian(std::tuple(8_c, 8_c, 3_c), std::tuple(1_c, 1_c));
    testee(domain, actual, in);

    for (int i = 1; i < 9; ++i)
        for (int j = 1; j < 9; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_DOUBLE_EQ(actual[i][j][k], expected(i, j, k)) << "i=" << i << ", j=" << j << ", k=" << k;
}
