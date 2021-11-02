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

constexpr auto lap = [](auto const &in) {
    return minus(multiplies(4_c, deref(in)),
        deref(shift<i, 1>(in)),
        deref(shift<j, 1>(in)),
        deref(shift<i, -1>(in)),
        deref(shift<j, -1>(in)));
};

template <auto D>
constexpr auto rdif = [](auto const &in) { return minus(deref(shift<D, 1>(in)), deref(in)); };

template <auto D>
constexpr auto ldif = [](auto const &in) { return minus(deref(in), deref(shift<D, -1>(in))); };

template <auto D>
constexpr auto flax = [](auto const &in, auto const &lap) {
    return lambda<[](auto in, auto res) { return if_(less(multiplies(in, res), 0_c), res, 0_c); }>(
        rdif<D>(in), rdif<D>(lap));
};

template <class Param>
constexpr auto hd = [](auto const &coeff, auto const &in) {
    return lambda<[](auto const &coeff, auto const &in, auto const &lap) {
        return minus(deref(in),
            multiplies(deref(coeff), plus(ldif<i>(Param::flax_i(in, lap)), ldif<j>(Param::flax_j(in, lap)))));
    }>(coeff, in, Param::lap(in));
};

template <bool Lap, bool FlaxI, bool FlaxJ>
struct param {
    static constexpr auto lap = lift<::lap, Lap>;
    static constexpr auto flax_i = lift<flax<i>, FlaxI>;
    static constexpr auto flax_j = lift<flax<j>, FlaxJ>;
};

template <class>
using lift_test = testing::Test;

using params_t = testing::Types<param<false, false, false>,
    param<false, true, false>,
    param<false, false, true>,
    param<false, true, true>,
    param<true, false, false>,
    param<true, true, false>,
    param<true, false, true>,
    param<true, true, true>>;

TYPED_TEST_SUITE(lift_test, params_t);

TYPED_TEST(lift_test, hd) {
    double actual[10][10][3] = {};
    double in[10][10][3];
    double coeff[10][10][3];
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 3; ++k) {
                in[i][j][k] = std::rand();
                coeff[i][j][k] = std::rand();
            }

    auto lap = [&](auto i, auto j, auto k) {
        return 4 * in[i][j][k] - in[i + 1][j][k] - in[i - 1][j][k] - in[i][j + 1][k] - in[i][j - 1][k];
    };

    auto flaxi = [&](auto i, auto j, auto k) {
        auto res = lap(i + 1, j, k) - lap(i, j, k);
        return res * (in[i + 1][j][k] - in[i][j][k]) < 0 ? res : 0;
    };

    auto flaxj = [&](auto i, auto j, auto k) {
        auto res = lap(i, j + 1, k) - lap(i, j, k);
        return res * (in[i][j + 1][k] - in[i][j][k]) < 0 ? res : 0;
    };

    auto expected = [&](auto i, auto j, auto k) {
        return in[i][j][k] -
               coeff[i][j][k] * (flaxi(i, j, k) - flaxi(i - 1, j, k) + flaxj(i, j, k) - flaxj(i, j - 1, k));
    };

    using stage_t = make_stage<hd<TypeParam>, std::identity{}, 0, 1, 2>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = cartesian(std::tuple(6_c, 6_c, 3_c), std::tuple(2_c, 2_c));
    testee(domain, actual, coeff, in);

    for (int i = 2; i < 8; ++i)
        for (int j = 2; j < 8; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_DOUBLE_EQ(actual[i][j][k], expected(i, j, k)) << "i=" << i << ", j=" << j << ", k=" << k;
}
