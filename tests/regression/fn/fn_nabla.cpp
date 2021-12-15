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

#include "simple_mesh.hpp"

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

constexpr auto zavg_helper = [](auto const &tmp, auto const &s) {
    return make_tuple(multiplies(tmp, tuple_get<0>(s)), multiplies(tmp, tuple_get<1>(s)));
};

template <auto E2V>
constexpr auto zavg = [](auto const &pp, auto const &s) {
    // auto tmp = sum(shift<E2V>(pp)) / 2;
    // auto ss = deref(s);
    // return std::tuple {
    //   tmp * std::get<0>(ss),
    //   tmp * std::get<1>(ss)
    // };
    return lambda<zavg_helper>(divides(sum(shift<E2V>(pp)), 2_c), deref(s));
};

constexpr auto nabla_helper = [](auto tmp, auto vol) {
    return make_tuple(divides(tuple_get<0>(tmp), vol), divides(tuple_get<1>(tmp), vol));
};

template <auto E2V, auto V2E, bool UseTmp>
constexpr auto nabla = [](auto const &pp, auto const &s, auto const &sign, auto const &vol) {
    // auto tmp = tuple_dot(shift<V2E>(lift<zavg<E2V>>(pp, s)), deref(sign));
    // auto v = deref(vol);
    // return std::tuple { std::get<0>(tmp) / v, std::get<1>(tmp) / v };
    return lambda<nabla_helper>(tuple_dot(shift<V2E>(lift<zavg<E2V>, UseTmp>(pp, s)), deref(sign)), deref(vol));
};

using namespace simple_mesh;
constexpr auto K = 3_c;
constexpr auto n_v2e = neighbours_num<decltype(v2e)>::value;

using params_t = testing::Types<std::false_type, std::true_type>;

template <class>
using lift_test = testing::Test;

TYPED_TEST_SUITE(lift_test, params_t);

TYPED_TEST(lift_test, nabla) {
    double pp[n_vertices][K];
    for (auto& ppp : pp)
        for (auto& p : ppp)
            p = rand() % 100;

    std::array<int, n_v2e> sign[n_vertices];
    for (auto &&ss : sign)
        for (auto &s : ss)
            s = rand() % 2 ? 1 : -1;

    double vol[n_vertices];
    for (auto &v : vol)
        v = rand() % 2 + 1;

    std::tuple<double, double> s[n_edges][K];
    for (auto& ss : s)
        for (auto& sss : ss)
            sss = {rand() % 100, rand() % 100};

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
            for (int j = 0; j != n_v2e; ++j) {
                auto edge = v2e[h][j];
                if (edge == -1)
                    break;
                res[i] += zavg(edge, v)[i] * sign[h][j];
            }
            res[i] /= vol[h];
        }
        return res;
    };

    std::tuple<double, double> actual[n_edges][K] = {};

    using stage_t = make_stage<nabla<e2v, v2e, TypeParam::value>, std::identity{}, 0, 1, 2, 3, 4>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = unstructured(std::tuple(n_vertices, K));
    testee(domain, actual, pp, s, sign, vol);

    for (int h = 0; h < n_vertices; ++h)
        for (int v = 0; v < K; ++v)
            tuple_util::for_each(
                [](auto actual, auto expected) { EXPECT_DOUBLE_EQ(actual, expected); }, actual[h][v], expected(h, v));
}
