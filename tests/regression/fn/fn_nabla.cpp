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

inline constexpr auto sum = reduce<plus, 0.>;
inline constexpr auto dot = reduce<[](auto acc, auto x, auto y) { return plus(acc, multiplies(x, y)); }, 0.>;

template <auto E2V>
inline constexpr auto zavg = lambda<[](auto pp, auto sx, auto sy) {
    return lambda<[](auto tmp, auto sx, auto sy) { return make_tuple(multiplies(tmp, sx), multiplies(tmp, sy)); }>(
        divides(sum(shift<E2V>(pp)), 2_c), deref(sx), deref(sy));
}>;

template <auto E2V, auto V2E>
inline constexpr auto nabla = lambda<[](auto pp, auto sx, auto sy, auto sign, auto vol) {
    return lambda<[](auto z, auto s, auto v) {
        return make_tuple(
            divides(dot(shift<V2E>(tuple_get<0>(z)), s), v), divides(dot(shift<V2E>(tuple_get<1>(z)), s), v));
    }>(ilift<zavg<E2V>>(pp, sx, sy), deref(sign), deref(vol));
}>;

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
    double sx[12][3];
    double sy[12][3];
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
        for (int v = 0; v < 3; ++v) {
            sx[h][v] = rand() % 100;
            sy[h][v] = rand() % 100;
        }

    auto zavg = [&](int h, int v) -> std::array<double, 2> {
        auto tmp = 0.;
        for (auto vertex : e2v[h])
            tmp += pp[vertex][v];
        tmp /= 2;
        return {tmp * sx[h][v], tmp * sy[h][v]};
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

    double actual_x[7][3] = {};
    double actual_y[7][3] = {};

    using stage_t = make_stage<nabla<e2v, v2e>, std::identity{}, meta::val<0, 1>, 2, 3, 4, 5, 6>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = unstructured(std::tuple(7_c, 3_c));
    testee(domain, actual_x, actual_y, pp, sx, sy, sign, vol);

    for (int h = 0; h < 7; ++h)
        for (int v = 0; v < 3; ++v) {
            auto exp = expected(h, v);
            EXPECT_DOUBLE_EQ(actual_x[h][v], exp[0]);
            EXPECT_DOUBLE_EQ(actual_y[h][v], exp[1]);
        }
}
