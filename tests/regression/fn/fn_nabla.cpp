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

using namespace gridtools;
using namespace fn;
using namespace literals;

constexpr auto zero = []<class T>(T) { return T{}; };
constexpr auto sum = reduce<plus, zero>;

constexpr auto tuple_dot_fun = [](auto acc, auto z, auto sign) {
    return make_tuple(plus(tuple_get<0>(acc), multiplies(tuple_get<0>(z), sign)),
        plus(tuple_get<1>(acc), multiplies(tuple_get<1>(z), sign)));
};
constexpr auto tuple_dot_init = [](auto z, auto sign) {
    return decltype(make_tuple(multiplies(tuple_get<0>(z), sign), multiplies(tuple_get<1>(z), sign))){};
};
constexpr auto tuple_dot = reduce<tuple_dot_fun, tuple_dot_init>;

template <auto E2V>
constexpr auto zavg = [](auto &&pp, auto &&s) {
    // auto tmp = sum(shift<E2V>(pp)) / 2;
    // auto ss = deref(s);
    // return std::tuple {
    //   tmp * std::get<0>(ss),
    //   tmp * std::get<1>(ss)
    // };
    return lambda<[](auto &&tmp, auto &&s) {
        return make_tuple(multiplies(tmp, tuple_get<0>(s)), multiplies(tmp, tuple_get<1>(s)));
    }>(divides(sum(shift<E2V>(std::forward<decltype(pp)>(pp))), 2_c), deref(std::forward<decltype(s)>(s)));
};

template <auto E2V, auto V2E>
constexpr auto nabla = [](auto &&pp, auto &&s, auto &&sign, auto &&vol) {
    // auto tmp = tuple_dot(shift<V2E>(ilift<zavg<E2V>>(pp, s)), deref(sign));
    // auto v = deref(vol);
    // return std::tuple { std::get<0>(tmp) / v, std::get<1>(tmp) / v };
    return lambda<[](auto &&tmp, auto &&vol) {
        return make_tuple(divides(tuple_get<0>(tmp), vol), divides(tuple_get<1>(tmp), vol));
    }>(tuple_dot(shift<V2E>(ilift<zavg<E2V>>(std::forward<decltype(pp)>(pp), std::forward<decltype(s)>(s))),
           deref(std::forward<decltype(sign)>(sign))),
        deref(std::forward<decltype(vol)>(vol)));
};

/*
 *          (2)
 *       1   2    3
 *   (1)  0     4   (3)
 *   11     (0)      5
 *   (6) 10      6  (4)
 *      9    8   7
 *          (5)
 */

constexpr std::array<int, 2> e2v[12] = {
    {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 0}, {3, 5}, {4, 0}, {4, 5}, {5, 0}, {5, 6}, {6, 0}, {6, 1}};

constexpr std::array<int, 6> v2e[7] = {{0, 2, 4, 6, 8, 10},
    {0, 1, 11, -1, -1, -1},
    {1, 2, 3, -1, -1, -1},
    {3, 4, 5, -1, -1, -1},
    {5, 6, 7, -1, -1, -1},
    {7, 8, 9, -1, -1, -1},
    {9, 10, 11, -1, -1, -1}};

TEST(fn, nabla) {

    double pp[7][3];
    std::tuple<double, double> s[12][3];
    std::array<int, 6> sign[7];
    double vol[7];

    for (int h = 0; h < 7; ++h) {
        for (int i = 0; i < 6; ++i)
            sign[h][i] = rand() % 2 ? 1 : -1;
        vol[h] = rand() % 2 + 1;
        for (int v = 0; v < 3; ++v)
            pp[h][v] = rand() % 100;
    }
    for (int h = 0; h < 12; ++h)
        for (int v = 0; v < 3; ++v)
            s[h][v] = {rand() % 100, rand() % 100};

    auto zavg = [&](int h, int v) -> std::array<double, 2> {
        auto tmp = 0.;
        for (auto vertex : e2v[h])
            tmp += pp[vertex][v];
        tmp /= 2;
        return {tmp * std::get<0>(s[h][v]), tmp * std::get<1>(s[h][v])};
    };

    auto expected = [&](int h, int v) {
        auto res = std::array{0., 0.};
        for (int i = 0; i != 2; ++i) {
            for (int j = 0; j != 6; ++j) {
                auto edge = v2e[h][j];
                if (edge == -1)
                    break;
                res[i] += zavg(edge, v)[i] * sign[h][j];
            }
            res[i] /= vol[h];
        }
        return res;
    };

    std::tuple<double, double> actual[7][3] = {};

    using stage_t = make_stage<nabla<e2v, v2e>, std::identity{}, 0, 1, 2, 3, 4>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = unstructured(std::tuple(7_c, 3_c));
    testee(domain, actual, pp, s, sign, vol);

    for (int h = 0; h < 7; ++h)
        for (int v = 0; v < 3; ++v) {
            auto exp = expected(h, v);
            auto act = actual[h][v];
            EXPECT_DOUBLE_EQ(std::get<0>(act), exp[0]);
            EXPECT_DOUBLE_EQ(std::get<1>(act), exp[1]);
        }
}
